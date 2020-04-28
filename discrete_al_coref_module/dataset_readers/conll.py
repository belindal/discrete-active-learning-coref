import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set
import torch

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans

from discrete_al_coref_module.dataset_readers.pair_field import PairField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotatated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


@DatasetReader.register("al_coref")
class HeldOutSetConllCorefReader(DatasetReader):
    # TODO FIX COMMENTS
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 simulate_user_inputs: bool = False,
                 fully_labelled_threshold: int = 1000,
                 saved_data_file: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._simulate_user_inputs = simulate_user_inputs
        # threshold for how many documents should be fully labelled (all remaining documents are half-labelled)
        self._fully_labelled_threshold = fully_labelled_threshold
        # serialized labels
        self._saved_labels = None
        if saved_data_file is not None:
            self._saved_labels = torch.load(saved_data_file)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        i = 0
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens,
                                              end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)

            percent_user_spans = 0.0
            if self._simulate_user_inputs and i >= self._fully_labelled_threshold:
                percent_user_spans = 1.0

            i += 1

            yield self.text_to_instance([s.words for s in sentences], sentences[0].document_id, sentences[0].sentence_id,
                                        canonical_clusters, percent_user_spans)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         document_id: str,
                         sentence_id: int,
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
                         user_threshold: Optional[float] = 0.0) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        document_id : ``str``, required.
            A string representing the document ID.
        sentence_id : ``int``, required.
            An int representing the sentence ID.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.
        user_threshold: ``Optional[float]``, optional (default = 0.0)
            approximate % of gold labels to label to hold out as user input.
            EX = 0.5, 0.33, 0.25, 0.125

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        flattened_sentences = [self._normalize_word(word)
                               for sentence in sentences
                               for word in sentence]

        metadata: Dict[str, Any] = {"original_text": flattened_sentences, "ID": document_id + ";" + str(sentence_id)}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters
            metadata["num_gold_clusters"] = len(gold_clusters)

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        user_threshold_mod = int(1 / user_threshold) if self._simulate_user_inputs and user_threshold > 0 else 0
        cluster_dict = {}
        simulated_user_cluster_dict = {}

        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for i in range(len(cluster)):
                    # use modulo to have a relatively even distribution of user labels across length of document,
                    # (since clusters are sorted)--so user simulated clusters are spread evenly across document
                    if user_threshold_mod == 0 or i % user_threshold_mod != user_threshold_mod - 1:
                        cluster_dict[tuple(cluster[i])] = cluster_id
                    simulated_user_cluster_dict[tuple(cluster[i])] = cluster_id

        # Note simulated_user_cluster_dict encompasses ALL gold labels, including those in cluster_dict
        # Consequently user_labels encompasses all gold labels
        spans: List[Field] = []
        if gold_clusters is not None:
            span_labels: Optional[List[int]] = []
            user_labels: Optional[List[int]] = [] if self._simulate_user_inputs and user_threshold > 0 else None
        else:
            span_labels = user_labels = None

        # our must-link and cannot-link constraints, derived from user labels
        # using gold_clusters being None as an indicator of whether we're running training or not
        must_link: Optional[List[int]] = [] if gold_clusters is not None else None
        cannot_link: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        doc_info = None
        if self._saved_labels is not None and metadata['ID'] in self._saved_labels:
            doc_info = self._saved_labels[metadata['ID']]
            span_labels = doc_info['span_labels'].tolist()
            if 'must_link' in doc_info:
                must_link = doc_info['must_link'].squeeze(-1).tolist()
                cannot_link = doc_info['cannot_link'].squeeze(-1).tolist()
        for sentence in sentences:
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if span_labels is not None:
                    if doc_info is None:
                        # only do if we haven't already loaded span labels
                        if (start, end) in cluster_dict:
                            span_labels.append(cluster_dict[(start, end)])
                        else:
                            span_labels.append(-1)
                    if self._simulate_user_inputs and user_threshold > 0:
                        if (start, end) in simulated_user_cluster_dict:
                            user_labels.append(simulated_user_cluster_dict[(start, end)])
                        else:
                            user_labels.append(-1)

                spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence)

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field}

        if must_link is not None and len(must_link) > 0:
            must_link_field = []
            cannot_link_field = []
            for link in must_link:
                must_link_field.append(PairField(
                    IndexField(link[0], span_field),
                    IndexField(link[1], span_field),
                ))
            for link in cannot_link:
                cannot_link_field.append(PairField(
                    IndexField(link[0], span_field),
                    IndexField(link[1], span_field),
                ))
            must_link_field = ListField(must_link_field)
            cannot_link_field = ListField(cannot_link_field)
            fields["must_link"] = must_link_field
            fields["cannot_link"] = cannot_link_field

        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)
            if user_labels is not None:
                fields["user_labels"] = SequenceLabelField(user_labels, span_field)

        if doc_info is not None:
            assert (
                fields["span_labels"].as_tensor(fields["span_labels"].get_padding_lengths()) != doc_info['span_labels']
            ).nonzero().size(0) == 0
            if 'must_link' in doc_info:
                assert 'must_link' in fields
                assert (
                    fields["must_link"].as_tensor(fields["must_link"].get_padding_lengths()) != doc_info['must_link']
                ).nonzero().size(0) == 0
                assert (
                    fields["cannot_link"].as_tensor(fields["cannot_link"].get_padding_lengths()) != doc_info['cannot_link']
                ).nonzero().size(0) == 0
        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
