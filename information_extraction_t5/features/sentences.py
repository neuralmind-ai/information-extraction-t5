"""Auxiliar methods for post-processing T5 input/output sentences."""
import re
from typing import List, Tuple, Union

from information_extraction_t5.features.questions.type_map import TYPE_TO_TYPENAME, COMPLEMENT_TYPE

SENTENCE_ID_PATTERN = r'\[SENT(.*?)\]'
SUBANSWER_PATTERN = r'([^[\]]+)(?:$|\[)'
TYPE_NAME_PATTERN = r'\[([A-Za-záàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑºª_ \/]*?)\]'

SENT_TOKEN = ' [SENT{}] '
T5_RAW_CONTEXT = str

# Type of a sentence that may have T5 identification tokens
# Example: '[SENT1] [Comarca] Campinas'
T5_SENTENCE = str


def _has_text(string: str) -> bool:
    """Returns True if a string has non whitespace text."""
    string_without_whitespace = string.strip()
    return len(string_without_whitespace) > 0


def _clean_sub_answer(sub_answer: str) -> str:
    """Removes undesired characters from a sub answer.

    Removes any `:` and whitespace the subanswer.
    """
    sub_answer = sub_answer.replace(':', '')
    sub_answer = sub_answer.strip()

    return sub_answer


def find_sub_answers(prediction_str: T5_SENTENCE) -> List[str]:
    """Returns a list containing the sub answers of a T5 sentence in the order
    they appear.

    Examples:
        >>> sentence = '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI'
        >>> sub_answers = _find_sub_answers(sentence)
        >>> print(sub_answers)
        ['Rua', 'PEDRO BIAGI']
    """
    sub_answer_list = []
    for sub_answer in re.findall(SUBANSWER_PATTERN, prediction_str):
        if _has_text(sub_answer):
            sub_answer = _clean_sub_answer(sub_answer)
            sub_answer_list.append(sub_answer)

    return sub_answer_list


def find_ids_of_sent_tokens(sentence: T5_SENTENCE) -> List[int]:
    """Returns a list containing the IDs of the SENT tokens if a T5 sentence in
    the order they appear.

    The ID is the number that follows a SENT token.

    Examples:
        >>> sentence = '[SENT1] Campinas'
        >>> ids = _find_ids_of_sent_tokens(sentence)
        >>> print(ids)
        [1]
    """
    ids = []
    for sentid in re.findall(SENTENCE_ID_PATTERN, sentence):
        try:
            ids.append(int(sentid))
        except:
            ids.append(sentid)

    return ids


def _convert_name_from_t5_to_type_name(name: str) -> str:
    """Converts name outputed by T5 for a type name.

    When the model was trained it learned to output display names from chunks.
    This method replaces the display names with their type name version.
    """
    if name not in TYPE_TO_TYPENAME:
        raise ValueError(f'Unknown type name: {name}')

    return TYPE_TO_TYPENAME[name]


def find_type_names(sentence: T5_SENTENCE, map_type: bool = True) -> List[str]:
    """Returns a list containing the names of the type tokens of a T5 sentence
    in the order they appear.

    The name is the text that appears inside the type token.

    Examples:
        >>> sentence = '[Logradouro] Campinas'
        >>> type_names = _find_type_names(sentence)
        >>> print(type_names)
        ['Logradouro']
    """
    type_names = re.findall(TYPE_NAME_PATTERN, sentence)
    if map_type:
        type_names = [
            _convert_name_from_t5_to_type_name(name) for name in type_names
        ]

    return type_names


def split_context_into_sentences(
        context: T5_RAW_CONTEXT
) -> List[str]:
    """Splits a question context into multiple questions.

    The criteria of splitting is simply every linebreak found.
    """
    return context.split('\n')


def split_t5_sentence_into_components(
        sentence: T5_SENTENCE,
        map_type: bool = True
) -> Tuple[List[int], List[str], List[str]]:
    """Splits the string outputed by T5 into its components.

    If no occurrences are found of a component, returns an empty list for it.

    Components:
        - sent ids: the ID that follows a SENT token.
        - type names: the name inside a answer type token.
        - sub answers: each answer fragment found.
    Args:
        sentence: a T5 output sentence.

    Examples:
        >>> sentence = '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI [SENT26] [Número]: 462 [SENT25] [Cidade]: Sertãozinho [SENT0] [Estado]: SP'
        >>> sent_ids, type_names, sub_answers = \
        >>>     split_t5_sentence_into_components(sentence)
        >>> print(sent_ids)
        [25, 25, 26 25, 0]
        >>> print(type_names)
        ['tipo_de_logradouro', 'logradouro', 'numero', 'cidade', 'estado']
        >>> print(sub_answers)
        ['Rua', 'PEDRO BIAGI', '462', 'Sertãozinho', 'SP'])

    Returns:
        Sentence ids, type names, answers/sub-answers
    """
    sent_ids = find_ids_of_sent_tokens(sentence)
    type_names = find_type_names(sentence, map_type=map_type)
    sub_answers = find_sub_answers(sentence)

    return sent_ids, type_names, sub_answers


def check_sent_id_is_valid(
        context: T5_RAW_CONTEXT, sent_id: int
) -> bool:
    """Returns True if a SENT ID is valid.

    An ID is valid when it corresponds to the ID of a sentence or its ID is 0.
    """
    if sent_id < 0:
        return False

    sentences = split_context_into_sentences(context)

    if len(sentences) < sent_id:
        return False

    return True
    

def deconstruct_answer(
    answer_sentence: T5_SENTENCE = ''
) -> Tuple[List[T5_SENTENCE], List[str]]:
    """Gets individual answer subsentences from the compound answer sentence.

    Args:
        answer sentence: a T5 output sentence.

    Examples:
        >>> sentence = '[SENT25] [Tipo de Logradouro]: Rua [SENT25] [Logradouro]: PEDRO BIAGI [SENT26] [Número]: 462 [SENT25] [Cidade]: Sertãozinho [SENT0] [Estado]: SP [aparece no texto] s paulo'
        >>> sub_sentences, type_names = deconstruct_answer(sentence)
        >>> print(sub_sentences)
        [
            '[SENT25] [tipo_de_logradouro] Rua', 
            '[SENT25] [logradouro] PEDRO BIAGI',
            '[SENT26] [numero] 462',
            '[SENT25] [cidade] Sertãozinho',
            '[SENT0] [estado] SP [aparece no texto] s paulo'
        ]
        >>> print(type_names)
        ['tipo_de_logradouro', 'logradouro', 'numero', 'cidade', 'estado']
        
    Returns:
        sub-ansers and type-names
    """
    sent_ids, type_names, sub_answers = split_t5_sentence_into_components(answer_sentence)
    sub_sentences = []
    all_type_names = []

    while len(sub_answers) > 0:
        sub_sentence = '' 

        if len(sent_ids) > 0:
            sent_id = sent_ids.pop(0)
            sentence_token = SENT_TOKEN.format(sent_id).strip()
            sub_sentence += sentence_token + ' '

        if len(type_names) > 0:
            type_name = type_names.pop(0)
            sub_sentence += f'[{type_name}]: '
            all_type_names.append(type_name)

        sub_answer = sub_answers.pop(0)
        sub_sentence += f'{sub_answer} '

        if len(type_names) > 0 and len(sub_answers) > 0 and type_names[0] == COMPLEMENT_TYPE:
            type_name = type_names.pop(0)
            sub_answer = sub_answers.pop(0)

            sub_sentence += f'[{type_name}] {sub_answer} '

        sub_sentences.append(sub_sentence.strip())

    return sub_sentences, all_type_names


def get_subanswer_from_subsentence(subsentence: T5_SENTENCE) -> T5_SENTENCE:
    """Get only the sub-answer from the current subsentence.

    Args:
        subsentence: a T5 subsentence.

    Examples:
        >>> subsentence = [SENT1] [no_da_matricula] 88975 [aparece no texto] 88.975
        >>> subanswer = get_subanswer_from_subsentence(subsentence)
        >>> print(subanswer)
        [no_da_matricula]: 88975
        
    Returns:
        subanswer that corresponds to subsentence without SENT_TOKEN and COMPLEMENT_TYPE

    """
    _, tn, ans = split_t5_sentence_into_components(subsentence, map_type=False)

    if len(ans) == 0:
        return ''

    if len(tn) == 0:
        subanswer = ans[0]
    else:
        subanswer = f'[{tn[0]}]: {ans[0]}'

    return subanswer


def get_raw_answer_from_subsentence(subsentence: T5_SENTENCE) -> Union[str, None]:
    """Get only the raw-text answer from the current subsentence.

    Args:
        subsentence: a T5 subsentence.

    Examples:
        >>> subsentence = [SENT1] [no_da_matricula] 88975 [aparece no texto] 88.975
        >>> subanswer = get_raw_answer_from_subsentence(subsentence)
        >>> print(subanswer)
        88.975
        
    Returns:
        subanswer that corresponds to subsentence without SENT_TOKEN and COMPLEMENT_TYPE

    """
    try:
        return subsentence.split(f'[{COMPLEMENT_TYPE}]')[1].strip()
    except:
        return None


def get_clean_answer_from_subanswer(subanswer: T5_SENTENCE) -> List[str]:
    """Get the final and pure answer from each sub-answer.

    Args:
        subanswer: subanswer extracted with function get_subanswer_from_subsentence.

    Examples:
        >>> subanswer = '[no_da_matricula]: 88975'
        >>> answer_ = get_clean_answer_from_subanswer(subanswer)
        >>> print(answer)
        ['88975']
        
    Returns:
        clean answers without the clues in square brackets
    """
    try:
        return find_sub_answers(subanswer)
    except:
        return ['']
