"""Utility methods to post-process model output."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from information_extraction_t5.features.sentences import (
    T5_SENTENCE,
    find_ids_of_sent_tokens,
    deconstruct_answer,
    get_raw_answer_from_subsentence,
    get_subanswer_from_subsentence
)


def group_qas(document_or_example_ids: List[str], group_by_typenames=True) -> Dict[str, List[int]]:
    """Groups the sentences according to qa-ids of the examples or documents.

    Args:
        sentences: List of qa-ids (strings)

    Returns:
        Dict with qa-ids (document-type + type-name) as keys and list of indexes
        of grouped sentences as values.
    """
    qid_dict = {}
    for idx, document_or_example_id in enumerate(document_or_example_ids):
        # When grouping by example_ids (pattern document_class.typename), add only the project 
        # (document class), such as matriculas, certidoes, etc. Ex.: qid_dict['matriculas'] = [0, 1, 2].
        # Must include only for original answers, excluding the ones related to dismembered sub-answers.
        if group_by_typenames and '~' not in document_or_example_id:  # '~' appears in sub-answers of originally compound answers
            proj = document_or_example_id.split('.')[0]
            if proj in qid_dict.keys():
                qid_dict[proj].append(idx)
            else:
                qid_dict[proj] = [idx]

        if document_or_example_id in qid_dict.keys():
            qid_dict[document_or_example_id].append(idx)
        else:
            qid_dict[document_or_example_id] = [idx]

        # multiple chunks are suffixed with _i. Here we group those cases removing the suffix.
        if group_by_typenames:
            comp = None
            try:
                document_or_example_id, comp = document_or_example_id.rsplit('~', 1)
            except:
                pass

            try:
                doc_ex_id, t = document_or_example_id.rsplit('_', 1)
                has_asterisk = t.endswith('*')
                if comp is None:
                    if has_asterisk:
                        t = t[:-1]
                t = int(t.strip())  # try to convert suffix to integer
                if comp is not None:
                    doc_ex_id += '~' + comp
                elif has_asterisk:
                    doc_ex_id += '*'
                    
                if doc_ex_id in qid_dict.keys():
                    qid_dict[doc_ex_id].append(idx)
                else:
                    qid_dict[doc_ex_id] = [idx]
            except:
                pass

    return qid_dict


def split_compound_labels_and_predictions(
    labels: List[T5_SENTENCE], predictions: List[T5_SENTENCE], document_ids: List[str],
    example_ids: List[str], probs: List[float], window_ids: List[str], keep_original_compound: bool = True,
    keep_disjoint_compound: bool = True
    ) -> Tuple[List[T5_SENTENCE], List[T5_SENTENCE], List[str], List[str],
               List[float], List[int], List[int], List[str], List[int], Dict]:
    """Splits compound answers as individual subsentences (complete sub-anwers) 
    like \"[SENT1] [Estado]: SP [aparece no texto]: São Paulo\" extending 
    original label and prediction sets.

    This is useful in inference for getting individual metrics for each
    subsentence that composes a compound answer.

    The function keeps for predictions only the first occurrence of the
    type-names that compose the labels. If the prediction has some type-name
    that is absent in the label, it is ignored. If the prediction has 
    sentence-id or raw-text, but the label does not have, those terms are
    considered as part of prediction, and certainly will result in misprediction.

    If keep_original_compound, the function returns the indices of the original
    sentences, ignoring the ones the reference individual subsentences. This is
    useful for getting metrics only for the answers as returned by the model.
    The metrics will appear as 'ORIG' in the metrics file.

    If keep_disjoint_compound, the function returns, for each document class,
    (1) the indices of non-compound answers and (2) the indices of subsentences
    of compound answers (ignoring the compound answers). In both cases, the 
    indices references the senteces with sentence-ids and raw-text complements
    already filtered. This is useful for getting metrics for each document class
    that can be compared with other experiments that does not use compound qas 
    and/or does not use sentence-ids and raw-text complements. The metrics will
    appear prefixed by 'DISJOINT_' in the metrics file.

    Examples:
        >>> labels = ['[SENT1] [Tipo de Logradouro]: Rua [SENT1] [Logradouro]: Abert Einstein']
        >>> predictions = ['[SENT1] [Tipo de Logradouro]: Rua [SENT1] [Logradouro]: 41bert Ein5tein [SENT1] [Bairro]: Cidade Universitária']
        >>> labels, predictions, document_ids, example_ids, probs, window_ids, sent_ids, raw_texts, _, _ = \
        >>>     split_compound_labels_and_predictions(labels, predictions, ['doc_1'], ['matriculas.endereco'], [0.98], ['1 1'])
        >>> print(labels)
        ['[SENT1] [tipo_de_logradouro]: Rua [SENT1] [fp_logradouro]: Abert Einstein', '[SENT1] [tipo_de_logradouro]: Rua', '[tipo_de_logradouro]: Rua', '[SENT1] [fp_logradouro]: Abert Einstein', '[fp_logradouro]: Abert Einstein']
        >>> print(predictions)
        ['[SENT1] [tipo_de_logradouro]: Rua [SENT1] [fp_logradouro]: 41bert Ein5tein [SENT1] [fp_bairro]: Cidade Universitária', '[SENT1] [tipo_de_logradouro]: Rua', '[tipo_de_logradouro]: Rua', '[SENT1] [fp_logradouro]: 41bert Ein5tein', '[fp_logradouro]: 41bert Ein5tein']
        >>> print(document_ids)
        ['doc_1', 'doc_1', 'doc_1', 'doc_1', 'doc_1']
        >>> print(example_ids)
        ['matriculas.endereco', 'matriculas.endereco~tipo_de_logradouro', 'matriculas.endereco~tipo_de_logradouro*', 'matriculas.endereco~fp_logradouro', 'matriculas.endereco~fp_logradouro*']
        >>> print(probs)
        [0.98, 0.0, 0.0, 0.0, 0.0]
        >>> print(window_ids)
        [[1, 1], [1], [1], [1], [1]]
        >>> print(sent_ids)
        [None, None, [1], None, [1]]
        >>> print(raw_texts)
        [None, None, None, None, None]
        
    Returns:
        labels, predictions, document_ids, example_ids, probs, window_ids, sent_ids, raw_texts
    """
    labels_new, predictions_new = [], []
    document_ids_new, example_ids_new, probs_new, window_ids_new, sent_ids, raw_texts = \
        [], [], [], [], [], []
    original_idx = []
    disjoint_answer_idx_by_doc_class = {}

    for label, prediction, doc_id, ex_id, prob, window_id in zip(
        labels, predictions, document_ids, example_ids, probs, window_ids):
        window_id = [ int(w) for w in window_id.split(' ') ]
        label_subsentences, label_type_names = deconstruct_answer(label)
        prediction_subsentences, prediction_type_names = deconstruct_answer(prediction)

        # this is not compound answer, then get the original label/predicion pair
        if len(label_type_names) <= 1 or keep_original_compound:
            label = ' '.join(label_subsentences)
            prediction = ' '.join(prediction_subsentences)

            labels_new.append(label)
            predictions_new.append(prediction)
            document_ids_new.append(doc_id)
            example_ids_new.append(ex_id)
            probs_new.append(prob)
            window_ids_new.append(window_id)
            sent_ids.append(None)
            raw_texts.append(None)

            # indexes to compute the f1 and exact ONLY with original (non-splitted) answers
            if keep_original_compound:
                idx = len(labels_new) - 1
                original_idx.append(idx)

            if len(label_type_names) <= 1:
                # remove sent-id and raw-text complement, if the label has,
                # in order to get metric only for the response per se.
                label_sa = get_subanswer_from_subsentence(label)
                pred_sa = get_subanswer_from_subsentence(prediction)
                    
                raw_text = get_raw_answer_from_subsentence(prediction_subsentences[0])
                sent_id = find_ids_of_sent_tokens(prediction_subsentences[0])

                ex_id_ = ex_id + '*'
                
                labels_new.append(label_sa)
                predictions_new.append(pred_sa)
                document_ids_new.append(doc_id)
                example_ids_new.append(ex_id_)
                probs_new.append(prob)
                window_ids_new.append(window_id)
                sent_ids.append(sent_id)
                raw_texts.append(raw_text)

                # keep by-document-class the indices of non-compound answers.
                # The sent-id and raw-text complement are already filtered.
                if keep_disjoint_compound:
                    idx = len(labels_new) - 1
                    doc_class = ex_id.split('.')[0]
                    if doc_class in disjoint_answer_idx_by_doc_class.keys():
                        disjoint_answer_idx_by_doc_class[doc_class].append(idx)
                    else:
                        disjoint_answer_idx_by_doc_class[doc_class] = [idx]

        if len(label_type_names) > 1:
            window_id = window_id[:1]  # for compound qa, the window_id is repeated
            for label_ss, label_tn in zip(label_subsentences, label_type_names):

                try:
                    # the same type-name was predicted, get the first occurrence
                    pred_idx = prediction_type_names.index(label_tn)
                    pred_ss = prediction_subsentences[pred_idx]
                except:
                    # the same type-name was not predicted, use empty
                    pred_ss = ''

                ex_id_ = ex_id + '~' + label_tn

                labels_new.append(label_ss)
                predictions_new.append(pred_ss)
                document_ids_new.append(doc_id)
                example_ids_new.append(ex_id_)
                probs_new.append(0.0)
                window_ids_new.append(window_id)
                sent_ids.append(None)
                raw_texts.append(None)
                
                # remove sent-id and raw-text complement, if the label has,
                # in order to get metric only for the response per se
                label_sa = get_subanswer_from_subsentence(label_ss)
                pred_sa = get_subanswer_from_subsentence(pred_ss)

                raw_text = get_raw_answer_from_subsentence(pred_ss)
                sent_id = find_ids_of_sent_tokens(pred_ss)

                ex_id_ = ex_id + '~' + label_tn + '*'

                labels_new.append(label_sa)
                predictions_new.append(pred_sa)
                document_ids_new.append(doc_id)
                example_ids_new.append(ex_id_)
                probs_new.append(0.0)
                window_ids_new.append(window_id)
                sent_ids.append(sent_id)
                raw_texts.append(raw_text)

                # keep by-document-class only indices of sub-responses for compound answers,
                # not the original compound answer.
                # The sent-id and raw-text complement are already filtered.
                if keep_disjoint_compound:
                    idx = len(labels_new) - 1
                    doc_class = ex_id.split('.')[0]
                    if doc_class in disjoint_answer_idx_by_doc_class.keys():
                        disjoint_answer_idx_by_doc_class[doc_class].append(idx)
                    else:
                        disjoint_answer_idx_by_doc_class[doc_class] = [idx]

    return labels_new, predictions_new, document_ids_new, example_ids_new, probs_new, \
        window_ids_new, sent_ids, raw_texts, original_idx, disjoint_answer_idx_by_doc_class


def get_highest_probability_window(
    labels: List[T5_SENTENCE], predictions: List[T5_SENTENCE], 
    document_ids: List[str], example_ids: List[str], probs: List[float],
    use_fewer_NA: bool = False
    ) -> Tuple[List[T5_SENTENCE], List[T5_SENTENCE], List[str], List[str], List[float], List[str]]:
    """Get the highest-probability components for each pair document, example.
    """
    if use_fewer_NA:
        na_cases = [ pred.count('N/A') for pred in predictions ]
        arr = np.vstack([np.array(labels), np.array(predictions),
                        np.array(document_ids), np.array(example_ids),
                        np.array(na_cases), np.array(probs, dtype=object)]).transpose()
        df1 = pd.DataFrame(arr, columns=['labels', 'predictions', 'document_ids', 'example_ids', 'na', 'probs'])
    else:
        arr = np.vstack([np.array(labels), np.array(predictions),
                        np.array(document_ids), np.array(example_ids),
                        np.array(probs, dtype=object)]).transpose()
        df1 = pd.DataFrame(arr, columns=['labels', 'predictions', 'document_ids', 'example_ids', 'probs'])

    # include windows-id to access which window got the highest probability.
    # In case of compound qa, the windows-id is replicated for each 
    # prediction subsentence
    df1['window_ids'] = df1.groupby(['document_ids', 'example_ids']).cumcount().astype(str)
    df1['window_ids'] = df1.apply(lambda x: ' '.join([x['window_ids']] * len(deconstruct_answer(x['predictions'])[0]) ), axis=1)

    if use_fewer_NA:
        # get the highest-probability sample among cases with fewer number if N/As
        # for each pair document-id / example-id
        df1 = df1.sort_values(['na', 'probs'], ascending=[True, False]).groupby(['document_ids', 'example_ids']).head(1)
        df1.sort_index(inplace=True)

        labels, predictions, document_ids, example_ids, _, probs, window_ids = df1.T.values.tolist()
    else:
        # get the highest-probability sample for each pair document-id / example-id
        df1 = df1.sort_values('probs', ascending=False).groupby(['document_ids', 'example_ids']).head(1)
        df1.sort_index(inplace=True)

        labels, predictions, document_ids, example_ids, probs, window_ids = df1.T.values.tolist()

    return labels, predictions, document_ids, example_ids, probs, window_ids
