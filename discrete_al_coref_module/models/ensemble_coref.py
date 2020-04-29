from typing import Dict, List, Any

from overrides import overrides
import torch

from allennlp.models.ensemble import Ensemble
from allennlp.models.model import Model
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.training.metrics import MentionRecall, ConllCorefScores
from allennlp.nn import util

from discrete_al_coref_module.models.coref import ALCoreferenceResolver


@Model.register("coref-ensemble")
class CorefEnsemble(Ensemble):
    def __init__(self, submodels: List[ALCoreferenceResolver]) -> None:
        super().__init__(submodels)
        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()

    def forward(self,
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                span_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                get_scores: bool = False,
                **kwargs,
            ) -> Dict[str, torch.Tensor]:
        num_models = len(self.submodels)
        mention_results = [submodel.get_mention_scores(text, spans) for submodel in self.submodels]

        # extract return values
        mask = mention_results[0]['mask']
        num_spans_to_keep = mention_results[0]['num_items_to_keep']
        text_mask = mention_results[0]['text_mask']
        all_mention_scores = torch.stack([mention_results[i]['scores'] for i in range(num_models)])
        # average across mention scores
        avg_mention_scores = all_mention_scores.mean(0)
        # ensure we don't select masked items
        avg_mention_scores = util.replace_masked_values(avg_mention_scores, mask, -1e20)
        # prune mentions with averaged scores
        _, top_span_indices_ensemble = avg_mention_scores.topk(num_spans_to_keep, 1)
        top_span_indices_ensemble, _ = torch.sort(top_span_indices_ensemble, 1)
        top_span_indices_ensemble = top_span_indices_ensemble.squeeze(-1)
        flat_top_span_indices_ensemble = util.flatten_and_batch_shift_indices(top_span_indices_ensemble, avg_mention_scores.size(1))
        top_span_mask = util.batched_index_select(mask, top_span_indices_ensemble, flat_top_span_indices_ensemble)
        top_span_mention_scores = util.batched_index_select(avg_mention_scores, top_span_indices_ensemble, flat_top_span_indices_ensemble)

        # feed averaged mention scores and top mentions back into model
        coref_scores_results = [submodel.get_coreference_scores(
            spans=spans,
            top_span_mention_scores=top_span_mention_scores,
            num_spans_to_keep=num_spans_to_keep,
            top_span_indices=top_span_indices_ensemble,
            flat_top_span_indices=flat_top_span_indices_ensemble,
            top_span_mask=top_span_mask,
            top_span_embeddings=util.batched_index_select(
                mention_results[i]['embeddings'], top_span_indices_ensemble, flat_top_span_indices_ensemble,
            ),
            text_mask=text_mask,
            get_scores=True,
        ) for i, submodel in enumerate(self.submodels)]

        # extract return values (should be the same)
        top_spans = coref_scores_results[0]['output_dict']['top_spans']
        valid_antecedent_indices = coref_scores_results[0]['output_dict']['antecedent_indices']
        valid_antecedent_log_mask = coref_scores_results[0]['ant_mask']
        all_coref_scores = torch.stack([coref_scores_results[i]['output_dict']['coreference_scores'] for i in range(num_models)])
        # average across coref scores
        avg_coref_scores = all_coref_scores.mean(0)
        # obtain predictions with averaged scores
        _, ensemble_predicted_antecedents = avg_coref_scores.max(2)	
        ensemble_predicted_antecedents -= 1

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": valid_antecedent_indices,
                       "predicted_antecedents": ensemble_predicted_antecedents}
        if get_scores:
            output_dict["coreference_scores"] = avg_coref_scores
            output_dict['top_span_indices'] = top_span_indices_ensemble
        
        # feed averaged coreference scores back to model
        # since just running evaluation, should be the same result regardless of submodel
        output_dict = self.submodels[0].score_spans_if_labels(
            output_dict=output_dict,
            span_labels=span_labels,
            metadata=metadata,
            top_span_indices=top_span_indices_ensemble,
            flat_top_span_indices=flat_top_span_indices_ensemble,
            top_span_mask=top_span_mask,
            top_spans=top_spans,
            valid_antecedent_indices=valid_antecedent_indices,
            valid_antecedent_log_mask=valid_antecedent_log_mask,
            coreference_scores=avg_coref_scores,
            predicted_antecedents=ensemble_predicted_antecedents,
        )

        self._mention_recall(output_dict['top_spans'], metadata)
        self._conll_coref_scores(output_dict['top_spans'], output_dict['antecedent_indices'],
                                 output_dict['predicted_antecedents'], metadata)
        if get_scores:
            output_dict['coreference_scores_models'] = all_coref_scores
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        return self.submodels[0].decode(output_dict)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)
        # also reset submodels
        for i in range(len(self.submodels)):
            self.submodels[i].get_metrics(reset)

        return {"coref_precision": coref_precision,
                "coref_recall": coref_recall,
                "coref_f1": coref_f1,
                "mention_recall": mention_recall}
