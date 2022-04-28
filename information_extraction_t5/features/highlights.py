from typing import Optional, Tuple, Union, Dict
from collections import OrderedDict

from fuzzysearch import find_near_matches
from fuzzywuzzy import process

from information_extraction_t5.features.sentences import (
    check_sent_id_is_valid,
    T5_RAW_CONTEXT,
    split_context_into_sentences,
)

estados = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'São Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins'
}

area = {
    'metro_quadrado':   ['m²', 'm2', 'metros quadrados'],
    'hectare':          ['has', 'hectares'],
    'alq_paulista':     ['alqueires paulistas', 'alqueires']
}


def include_variations(query):
    """Given a canonical format, include possible variations of how the 
    information can appear in the text.
    """
    if query in estados.keys():
        return [estados[query]]
    if query in area.keys():
        return area[query]
    return []


def find_sentence_of_sent_id(context: T5_RAW_CONTEXT, sent_id: int) -> str:
    """Returns the sentence of number `sent_id` in the context.

    This method assumes the sentence ids are defined by linebreaks and start at
    1.

    Args:
        context: question raw context
        sent_id: Index of the sentence. Must be greater or equal to 0.
    """
    assert sent_id >= 0, (
        f'SENT id must be greater or equal to 0. Received: {sent_id}')
    
    sentences = split_context_into_sentences(context)

    return sentences[sent_id - 1]


def find_indexes_of_sentence(
        context: T5_RAW_CONTEXT, sent_id: int
) -> Union[Tuple[int, int], Tuple[None, None]]:
    """Returns character indexes for the start and end of a sentence in the
    context.

    This method assumes the sentence ids are defined by linebreaks and start at
    1.
    """
    sentence = find_sentence_of_sent_id(context, sent_id)
    # get the start_char and end_char of the sentence
    start_char = context.find(sentence)
    end_char = context.find('\n', start_char)

    return start_char, end_char


def get_levenshtein_dist(
    query_string,
    levenshtein_dist_dict: Optional[Dict[int, int]] = None
) -> int:
    """Returns a maximum levenshtein distance based on query string length."""
    if levenshtein_dist_dict is None:
        levenshtein_dist_dict=OrderedDict({3: 0, 10: 1, 20: 3, 30: 5})
    for str_size, dist in levenshtein_dist_dict.items():
        if len(query_string) < str_size:
            return dist
    return list(levenshtein_dist_dict.values())[-1]


def fuzzy_extract(
        query_string: str, large_string: str, score_cutoff: int = 30,
        max_levenshtein_dist: Union[int, Dict[int, int]] = -1,
        verbose: bool = False
) -> Union[Tuple[int, int], Tuple[None, None]]:
    """Fuzzy matches query string (and its variations) on a large string.

    Args:
        query_string: substring to be searched inside another string.
        large_string: the string to be searched on.
        score_cutoff: fuzzy matches with a score below this one will be ignored.
        max_levenshtein_dist: if a Dict, then changes the maximum levenshtein
            distance of matches (value) based on `query_string` length (key).
            Otherwise, an int should be supplied for a fixed maximum distance.
        verbose: When True, prints debug messages to stdout.

    Returns:
         Indexes of the start and end characters of the best match. If nothing
         is found, returns (None, None) instead.
    """
    if max_levenshtein_dist == -1:
        OrderedDict({3: 0, 10: 1, 20: 3, 30: 5})
    query_strings = include_variations(query_string) + [query_string]
    matches = []
    starts = []
    ends = []
    scores = []
    large_string = large_string.lower()

    for query_string in query_strings:
        query_string = query_string.lower()
        if verbose:
            print(f'query: {query_string}')

        # set dynamic Levenshtein distance
        if isinstance(max_levenshtein_dist, dict):
            max_l_dist_query = get_levenshtein_dist(query_string, max_levenshtein_dist)
        else:
            max_l_dist_query = max_levenshtein_dist

        all_matches = process.extractBests(query_string, (large_string,),
                                           score_cutoff=score_cutoff)
        for large, _ in all_matches:
            if verbose:
                print('word::: {}'.format(large))
            for match in find_near_matches(query_string, large,
                                           max_l_dist=max_l_dist_query):
                matched = match.matched
                start = match.start
                end = match.end
                score = match.dist
               
                if verbose:
                    print(f"match: {matched}\tindex: {start}\tscore: {score}")

                matches.append(matched)
                starts.append(start)
                ends.append(end)
                scores.append(score)

    if len(matches) == 0:
        return None, None

    best_id = scores.index(min(scores))

    return starts[best_id], ends[best_id]


def get_answer_highlight(
        answer: str, sent_id: int, context: T5_RAW_CONTEXT,
        sentence_expansion: int = 0, verbose: bool = False
) -> Union[Tuple[int, int, str], Tuple[None, None, None]]:
    r"""Given a single answer and its SENT ID, returns highlights of its
    location within the context.

    Sometimes the answer has line breaks in the middle of it (ex.: São\nPaulo).
    To find the highlight even on these cases, this optionally expands the
    highlight window some sentences beyond the original ID.

    Args:
        answer: the answer to search on the context.
        sent_id: ID of the sentence the answer is in (or starts at).
        context: the question raw context.
        sentence_expansion: When this is 0, looks for the answer only in the
            sentence pointed by SENT ID. If the value is a `N` greater than 0,
            then looks for it on the `N` sequences that come after SENT ID,
            i.e. the interval `[SENT ID, ..., SENT ID + N]`.
        verbose: If True, enables debug prints.

    Examples:
        >>> answer = 'Rua Albert Einstein'
        >>> sent_id = 3
        >>> context = "Campinas\n\nRua 4lbert \nE1nstein 1000"
        >>> get_answer_highlight(answer, sent_id, context, sentence_expansion=2)
        fuzzy ==> answer: Rua Albert Einstein, sentence: "Rua 4lbert  E1nstein 1000"
        (10, 30, 'Rua 4lbert \nE1nstein')
    """
    sentence = find_sentence_of_sent_id(context, sent_id)

    expanded_sentence = [sentence]
    for i in range(1, sentence_expansion + 1):
        is_valid = check_sent_id_is_valid(context, sent_id + i)
        if not is_valid:
            break

        extra_sentence = find_sentence_of_sent_id(context, sent_id + i)
        expanded_sentence.append(extra_sentence)
    sentence = ' '.join(expanded_sentence)

    if verbose:
        print(f'fuzzy ==> answer: {answer}, sentence: "{sentence}"')

    shift, _ = find_indexes_of_sentence(context, sent_id)
    start_char, end_char = fuzzy_extract(answer, sentence)

    if start_char is None or end_char is None:
        highlight = None

    else:
        start_char += shift
        end_char += shift
        highlight = context[start_char:end_char]

    return start_char, end_char, highlight
