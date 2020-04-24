from typing import Dict, List, Any

from overrides import overrides
import torch

from allennlp.models.ensemble import Ensemble
from allennlp.models.model import Model
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.training.metrics import MentionRecall, ConllCorefScores
from allennlp.models.coreference_resolution import CoreferenceResolver
from allennlp.nn import util
import pdb


@Model.register("coref-ensemble")
class CorefEnsemble(Ensemble):
    def __init__(self, submodels: List[CoreferenceResolver]) -> None:
        super().__init__(submodels)
        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()

    def forward(self,
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                span_labels: torch.IntTensor = None,
                user_labels: torch.IntTensor = None,
                must_link: torch.IntTensor = None,
                cannot_link: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                get_scores: bool = False,
                #for_training: bool = True,
                #training_model: int = 0
                ) -> Dict[str, torch.Tensor]:
        num_models = len(self.submodels)
        mention_results = [submodel(text, spans, span_labels, user_labels, must_link, cannot_link, metadata, get_scores, return_mention_scores=True) for submodel in self.submodels]
        mask = mention_results[0]['mask']
        num_items_to_keep = mention_results[0]['num_spans_to_keep']
        text_mask = mention_results[0]['text_mask']
        all_mention_scores = torch.stack([mention_results[i]['mention_scores'] for i in range(num_models)])
        avg_mention_scores = all_mention_scores.mean(0)
        # ensure we don't select masked items
        avg_mention_scores = util.replace_masked_values(avg_mention_scores, mask, -1e20)
        _, top_indices = avg_mention_scores.topk(num_items_to_keep, 1)
        top_indices, _ = torch.sort(top_indices, 1)
        top_indices = top_indices.squeeze(-1)
        flat_top_indices = util.flatten_and_batch_shift_indices(top_indices, avg_mention_scores.size(1))
        top_mask = util.batched_index_select(mask, top_indices, flat_top_indices)
        top_scores = util.batched_index_select(avg_mention_scores, top_indices, flat_top_indices)
        top_spans_info = [{'mention_scores': mention_results[i]['mention_scores'], 'top_scores': top_scores,
                           'span_indices': top_indices, 'top_mask': top_mask, 'flat_top_indices': flat_top_indices,
                           'text_mask': text_mask, 'span_embeddings': mention_results[i]['embeds']}
                          for i in range(num_models)]

        # feed averaged mention scores and top mentions back into model
        ret_values = [submodel(text, spans, span_labels, user_labels, must_link, cannot_link, metadata, get_scores=True,
                               top_spans_info=top_spans_info[i], return_coref_scores=True)
                      for i, submodel in enumerate(self.submodels)]

        top_spans = ret_values[0]['output_dict']['top_spans']
        ant_inds = ret_values[0]['output_dict']['antecedent_indices']
        all_coref_scores = torch.stack([ret_values[i]['output_dict']['coreference_scores'] for i in range(num_models)])
        avg_coref_scores = all_coref_scores.mean(0)

        coref_scores_output_dict = {'top_spans': top_spans, 'antecedent_indices': ant_inds,
                                    'coreference_scores': avg_coref_scores}
        coref_scores_info = {'output_dict': coref_scores_output_dict, 'top_span_inds': ret_values[0]['top_span_inds'],
                             'top_span_mask': ret_values[0]['top_span_mask'],
                             'valid_antecedent_log_mask': ret_values[0]['ant_mask']}
        # feed averaged mention scores and other variables back into model
        # should produce the same results
        output_dict = self.submodels[0](text, spans, span_labels, user_labels, must_link, cannot_link, metadata,
                                        get_scores, coref_scores_info=coref_scores_info)
        # run other models so coref scores are at the same state
        for i in range(1, num_models):
            self.submodels[i](text, spans, span_labels, user_labels, must_link, cannot_link, metadata, get_scores,
                              coref_scores_info=coref_scores_info)

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
