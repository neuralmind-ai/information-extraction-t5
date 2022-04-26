"""Utility methods to preprocess model input."""
from collections import Counter
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

from information_extraction_t5.features.questions import (
    COMPLEMENT,
    QUESTION,
    QUESTION_DICT,
    QUESTIONS as ALL_QUESTIONS,
    SUBQUESTION_DICT,
)
from information_extraction_t5.features.questions.type_map import COMPLEMENT_TYPE
from information_extraction_t5.features.sentences import SENT_TOKEN

# Large number to not let the number of sentences be too large for a model.
MAX_SENTENCES = 9999


def _replace_brackets_with_parenthesis(text: str) -> str:
    text = text.replace('{', '(')
    text = text.replace('}', ')')

    return text


def _replace_linebreak_with_token_patterns(
        text: str, token_pattern: str = SENT_TOKEN
) -> Tuple[str, int]:
    """Returns new string with `\n` replaced with the token pattern and the
    number of tokens."""
    num_tokens = text.count('\n')
    text = text.replace('\n', token_pattern)

    return text, num_tokens


def _replace_linebreaks_with_tokens(text: str) -> str:
    r"""Replaces every `\n` in a string with a numbered SENT token.

    If the inputs string has brackets, they will be replaced with parenthesis.
    Always adds least one SENT token at the beginning of the new sentence.
    Tokens are numerated starting from 1.

    Args:
        text: string to have `\n` replaced. It can't be split into more than
            MAX_SENTENCES.

    Examples:
        >>> sentence = 'Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho\nSP'
        >>> new_sentence = _replace_linebreaks_with_tokens(sentence)
        >>> print(new_sentence)
        ' [SENT1] Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho [SENT2] SP'

    Returns:
        New string with token instead of `\n`
    """
    # Should have at least one SENT token at start
    text = '\n' + text
    text = _replace_brackets_with_parenthesis(text)
    text, num_tokens = _replace_linebreak_with_token_patterns(text)

    assert num_tokens <= MAX_SENTENCES, 'Maximum number of sentences violated.'

    # token numeration must start from 1
    text = text.format(*range(1, num_tokens + 1))

    return text


def _replace_linebreaks_with_spaces(text: str) -> str:
    r"""Replaces every `\n` in a string with a space.

    Examples:
        >>> sentence = 'Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho\nSP'
        >>> new_sentence = _replace_linebreaks_with_spaces(sentence)
        >>> print(new_sentence)
        'Rua PEDRO BIAGI 462 Apartamento nº 103, 1º Andar do RESIDENCIAL IMPERIAL. Sertãozinho SP'
    """
    text = text.replace('\n', ' ')

    return text


def _get_id_based_on_linebreaks(context: str, answer_position: int) -> int:
    """Recovers the sentence-id assuming the context is always partiotioned based
    on occurrences of linebreaks.

    Args:
        context: text context of the question
        answer_position: index of last character from answer.
    """
    if answer_position == -1:
        return 0

    sent_id = Counter(context[:answer_position])['\n'] + 1

    return sent_id


def get_questions_for_chunk(
        qa_id: str = 'matriculas.imovel.comarca', is_compound: bool = False,
        return_dict: bool = False, all_questions: QUESTION_DICT = ALL_QUESTIONS
        ) -> Union[List[QUESTION], QUESTION_DICT, SUBQUESTION_DICT]:
    """Returns a list of questions for a specific qa_id, or a dict mapping 
    typenames to question for building compound answers. The function can 
    return also all the questions.

    Args:
        qa_id: the id of a question-answer, generally represented by dot-separated
            document class, chunks typenames, and possibly subchunks typenames.
            use 'all' to get a dictionary containing all the possible questions.
        is_compound: if qa_id represents a compound field.
        return_dict: if the function should return a dict that is useful to build
            compound answers.
        all_questions: Dictionary with all the questions and subquestions.

    Examples:
        >>> questions = {'question1': {'subquestion1': ['What?']}}
        >>> get_questions_for_chunk('all', all_questions=questions)
        {'question1': {'subquestion1': ['What?']}}
        >>> get_questions_for_chunk('matriculas.question1', all_questions=questions)
        {'subquestion1': ['What?']}
    
    Returns:
        List of all questions for a specific field, or dictionary with all the questions
        for a compound field.
    """
    if qa_id == 'all':
        return all_questions

    typenames = qa_id.split('.')
    questions = all_questions
    for typename in typenames:
        questions = questions[typename]
    
    if is_compound:
        questions = questions['compound']

    assert isinstance(questions, List) != return_dict, (
        f'Shouldn\'t you set "is_compound=True" for the field {qa_id} to get a '
        'list of questions for a specific compound typename? Or set '
        '"return_signature=True" to get the ordered dict of typenames to build '
        'a compound answer?')

    return questions


def get_qa_ids_recursively(dict_or_list, base_qa_id, list_of_use_compound_question, 
    list_of_compound_chunks_to_ignore, list_of_subchunks_to_skip, qa_ids_list=[]):
    """Auxiliar function to get recursively all the possible qa_ids"""

    if isinstance(dict_or_list, List) and not base_qa_id.endswith('compound'):
        qa_ids_list.append(base_qa_id)

    if isinstance(dict_or_list, Dict) or isinstance(dict_or_list, OrderedDict):
        if base_qa_id in list_of_use_compound_question:
            qa_ids_list.append(base_qa_id)

        elif base_qa_id not in list_of_compound_chunks_to_ignore:
            for typename, value in dict_or_list.items():
                if typename not in list_of_subchunks_to_skip:
                    qa_id = f'{base_qa_id}.{typename}'
                    _ = get_qa_ids_recursively(
                        value.copy(), qa_id,
                        list_of_use_compound_question,
                        list_of_compound_chunks_to_ignore,
                        list_of_subchunks_to_skip, qa_ids_list)

    return qa_ids_list


def get_all_qa_ids(
        document_class: Optional[str] = None,
        list_of_type_names: List[str] = [],
        list_of_use_compound_question: List[str] = [],
        list_of_subchunks_to_list: List[str] = [],
        list_subchunks_to_complement_siblings: List[str] = [], 
        list_of_subchunks_to_skip: List[str] = [],
        all_questions: QUESTION_DICT = ALL_QUESTIONS
        ) -> List[str]:
    """Returns a list of all possible qa_ids that will be used to force 
    the qa, even if the chunk does not exist.

    Args:
        document_class: class of documents to extract qa_ids. Use None to
            get for all possible document classes.
        list_of_typenames: list of type-names.
        list_of_use_compound_question: list of compound qa_ids.
        list_of_subchunks_to_list: list of listing qa_ids.
        list_subchunks_to_complement_sibling_questions: list of subchunks that 
            will complement siblings, and does not require a qa.
        list_of_subchunks_to_skip: list of subchunks that will be skipped.

    Examples:
        >>> typenames = ['matriculas.imovel', 'matriculas.endereco', 'certidoes.resultado']
        >>> use_compound = ['matriculas.endereco']
        >>> get_all_qa_ids('matriculas', typenames, use_compound)
        ['matriculas.imovel.no_da_matricula', 'matriculas.imovel.oficio', 'matriculas.imovel.comarca', 
            'matriculas.imovel.estado', 'matriculas.endereco']
        
    Returns:
        List of all possible qa_ids.
    """
    all_qa_ids = []

    # ignore chunks for which one subchunk will complement siblings.
    # we cannot force qas, since the question depends on a information
    # that is possibly non-annotated.
    list_of_compound_chunks_to_ignore = [sc.rsplit('.', 1)[0] 
        for sc in list_subchunks_to_complement_siblings]

    for doc_class, questions_dict in all_questions.items():
        if document_class is not None and doc_class != document_class: continue

        for typename, list_or_dict in questions_dict.items():
            qa_id = f'{doc_class}.{typename}'
            
            if qa_id in list_of_type_names:
                qa_ids = get_qa_ids_recursively(list_or_dict, qa_id, list_of_use_compound_question, 
                    list_of_compound_chunks_to_ignore, list_of_subchunks_to_skip, [])
                for qa_id in qa_ids:
                    all_qa_ids.append(qa_id)

    # for listing qa_ids, keep only document-class and last subchunk
    # with the suffix "_list"
    for qa_id in list_of_subchunks_to_list:
        typenames = qa_id.split('.')
        if document_class is None or document_class == typenames[0]: 
            qa_id = f'{typenames[0]}.{typenames[-1]}_list'
            all_qa_ids.append(qa_id)

    return all_qa_ids


def complement_questions_to_require_rawdata(
        questions: Union[QUESTION, List[QUESTION]], complement: str = COMPLEMENT
) -> Union[QUESTION, List[QUESTION]]:
    """Add complementary text to a question or questions.

    This indicates to the model it must give a subanswer with part of the
    context's raw text.
    """
    if isinstance(questions, str):  # simple question
        questions = questions.replace('?', complement)
    if isinstance(questions, list):  # list of questions
        questions = [q.replace('?', complement) for q in questions]
    return questions
    

def complement_questions_with_information(
        questions: Union[QUESTION, List[QUESTION]], complement: List[str] = []
) -> Union[QUESTION, List[QUESTION]]:
    """Add complementary information to a specific question or questions.

    This is used to build specific questions.
    """
    if isinstance(questions, str):  # simple question
        questions = questions.format(*complement)
    if isinstance(questions, list):  # list of questions
        questions = [q.format(*complement) for q in questions]
    return questions


def generate_t5_input_sentence(
        context: str, question: str, use_sentence_id: bool
) -> str:
    """Returns a T5 input sentence based on a question and its context.

    Args:
        context: text context of the question
        question: the question
        use_sentence_id: if True, every newline on the context will be replaced
            by a SENT token. Otherwise they are replaced with spaces.
    """
    if use_sentence_id:
        context = _replace_linebreaks_with_tokens(context)
    else:
        context = _replace_linebreaks_with_spaces(context)

    t5_sentence = f'question: {question} context: {context}'
    return t5_sentence


def generate_t5_label_sentence(
        answer: str, answer_start: Union[List[int], int], context: str,
        use_sentence_id: bool
) -> str:
    """Returns a T5 label sentence for simple or compound answers.

    Args:
        answer: answer of the current questions
        answer_start: char position of answer starting
        context: text context of the question
        use_sentence_id: if True, every newline on the context will be replaced
            by a SENT token. Otherwise they are replaced with spaces.
    """
    if use_sentence_id:
        if isinstance(answer_start, list):
            # That is a compound_answer, like: "[Valor]: 500,00 [Unidade]: metro_quadrado"

            # Separate the compound answer in sub-answers: --, Valor] 500,00, Unidade] metro_quadrado
            # that could be problematic if some sub-answer has brackets, besides COMPLEMENT_TYPE
            sub_answers = answer.split('[')[1:] 
            token_pattern = SENT_TOKEN.strip()                    

            # Extract sentence-ids for each sub-answer
            sent_ids = []
            for sub_answer_start in answer_start:
                sent_ids.append(_get_id_based_on_linebreaks(context,
                                                            sub_answer_start))

            # Prepare the final answer with sentence-ids: "[SENTx] [Valor]: 500,00 [SENTy] [Unidade]: metro_quadrado"
            answer = ''
            for sub_answer in sub_answers:
                if sub_answer.startswith(COMPLEMENT_TYPE):
                    answer = f'{answer}[{sub_answer}'
                else:
                    answer = f'{answer}{token_pattern} [{sub_answer}'

            # Include the sentence-ids
            answer = answer.format(*sent_ids)
        elif isinstance(answer_start, int):
            # That is a simple answer

            sent_id = _get_id_based_on_linebreaks(context, answer_start)
            answer = f'[SENT{sent_id}] {answer}'
        else:
            # That is an occurrence of non-annotated data, as publicacoes (null in squad json)
            # [SENTX] is not included
            pass

    return answer
