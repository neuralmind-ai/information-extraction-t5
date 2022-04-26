"""Convert a simple JSON dataset into SQuAD format."""
from typing import Dict, List, Optional
from transformers import T5Tokenizer
import numpy.random as nr

from information_extraction_t5.features.context import get_context
from information_extraction_t5.features.questions.type_map import TYPENAME_TO_TYPE
from information_extraction_t5.features.preprocess import get_questions_for_chunk

WARNING_MISSING_TYPENAMES = []


def get_question_answers(document: Dict[str, str],
                         questions: Optional[List[str]] = None,
                         qa_id: str = 'publicacoes.instancia',
                         choose_question: str = 'first'):
    """Gets question-answers in SQUAD format for the specified type name.

    The answers encompass only the canonical response (value).
    The size of the list is:
    - zero: if there's no question-answer with the specified type name or if the
        corresponding value of a type name key in the document is not a string.
    - one: choose_question is 'first' or 'random'.
    - N: an element for each question passed to the questions parameter.

    Returns:
        List of dictionaries where each element is a question and its answers.
    """
    if questions is None:
        questions = []

    subanswer = document
    qa_id_split = qa_id.split('.')

    for type_name in qa_id_split[1:]:
        subanswer = subanswer[type_name]
    
    # select questions
    if choose_question == 'first':
        selected_questions = [questions[0]]
    elif choose_question == 'random':
        idx = nr.randint(len(questions))
        selected_questions = [questions[idx]]
    else:
        selected_questions = questions

    qas = []
    answer = f"[{TYPENAME_TO_TYPE[type_name]}]: {subanswer}"
    for question in selected_questions:
        answers = [ 
            {
                "answer_start": -1,  # None,
                "text": answer,
            }
        ]
        qa = {
            "answers": answers,
            "question": question,
            "id": qa_id,
        }
        qas.append(qa)
    return qas


def get_compound_question_answers(
        document: Dict[str, str],
        questions: Optional[List[str]] = None,
        qa_id: str = 'publicacoes.instancia_orgao_tipo',
        choose_question: str = 'first') -> List[Dict]:
    """Gets question-answers in SQUAD format for the specified type names.

    The answers encompass only the canonical response (value).
    The size of the list is:
    - zero: if there's no question-answer with the specified type name or if the
        corresponding value of a type name key in the document is not a string.
    - one: choose_question is 'first' or 'random'.
    - N: an element for each question passed to the questions parameter.

    Returns:
        List of dictionaries where each element is a question and its answers.
    """
    # select questions
    if questions is None:
        questions = []
    if choose_question == 'first':
        selected_questions = [questions[0]]
    elif choose_question == 'random':
        idx = nr.randint(len(questions))
        selected_questions = [questions[idx]]
    else:
        selected_questions = questions

    type_name = qa_id.split('.')[1]

    all_type_names = get_questions_for_chunk(qa_id=qa_id, return_dict=True).copy()
    for tn in all_type_names.keys():
        if tn == 'compound':
            continue
        all_type_names[tn] = f'[{TYPENAME_TO_TYPE[tn]}]: N/A'
    if 'compound' in all_type_names.keys():
        all_type_names.pop('compound')

    # preparing the compound answer
    for tn in document[type_name].keys():
        type = TYPENAME_TO_TYPE[tn]
        subanswer = document[type_name][tn]

        if tn in all_type_names.keys():
            all_type_names[tn] = f"[{type}]: {subanswer}"
        elif not tn in WARNING_MISSING_TYPENAMES:
            print(f'WARNING: type-name {tn} is not in question signature for {type_name}: please add it in the OrderedDict if you want to keep.')
            WARNING_MISSING_TYPENAMES.append(tn)

    answer = ' '.join(all_type_names.values())

    qas = []
    for question in selected_questions:
        answers = [ 
            {
                "answer_start": -1,  # None,
                "text": answer
            }
        ]
        qa = {
            "answers": answers,
            "question": question,
            "id": qa_id,
        }
        qas.append(qa)
    return qas


def get_notapplicable_question_answers(
    qa_id: str = 'matriculas.endereco',
    choose_question: str = 'first',
    list_of_use_compound_question: Optional[List[str]] = None):
    """
    Return a list of question-answers in SQUAD format for non-annotated
    type-names.

    The size of the list is:
    - one (choose_question as 'first' or 'random')
    - the number of questions defined as 'compound' for the current chunk
        returned by get_questions_for_chunk(chunk) (choose_question as 'all')
    """
    if list_of_use_compound_question is None:
        list_of_use_compound_question = []

    is_compound = qa_id in list_of_use_compound_question

    questions = get_questions_for_chunk(qa_id=qa_id, is_compound=is_compound)
    if questions is None:
        questions = []
    if choose_question == 'first':
        selected_questions = [questions[0]]
    elif choose_question == 'random':
        idx = nr.randint(len(questions))
        selected_questions = [questions[idx]]
    else:
        selected_questions = questions

    if is_compound:
        # type_name = qa_id.split('.')[1]
        all_type_names = get_questions_for_chunk(qa_id=qa_id, return_dict=True).copy()
        for tn in all_type_names.keys():
            if tn == 'compound':
                continue
            all_type_names[tn] = f'[{TYPENAME_TO_TYPE[tn]}]: N/A'
        if 'compound' in all_type_names.keys():
            all_type_names.pop('compound')
        
        answer = ' '.join(all_type_names.values())
    else:
        type_name = qa_id.split('.', 1)[1]
        type = TYPENAME_TO_TYPE[type_name]

        answer = f"[{type}]: N/A"

    qas = []
    for question in selected_questions:
        answers = [ 
            {
                "answer_start": -1,  # None,
                "text": answer
            }
        ]
        qa = {
            "answers": answers,
            "question": question,
            "id": qa_id,
        }
        qas.append(qa)
    return qas


def get_document_data(document: Dict,
                      document_type: str = 'publicacoes',
                      all_qa_ids: List[str] = ['publicacoes.orgao'],
                      max_size: int = 4000,
                      list_of_use_compound_question: Optional[List[str]] = None,
                      list_of_type_names: Optional[List[str]] = None,
                      context_content: str = 'abertura',
                      window_overlap: float = 0.5,
                      max_windows: int = 3,
                      tokenizer: T5Tokenizer = None,
                      max_tokens: int = 512,
                      choose_question: str = 'first',
                      use_sentence_id: bool = False):
    # using the document uuid as title
    # paragraphs will contain only one dict with context of document and all the
    # question-answers
    if list_of_type_names is None:
        list_of_type_names = []
    if list_of_use_compound_question is None:
        list_of_use_compound_question = []

    # assuming that this is the largest question
    largest_question = 'Quais são as principais informações do documento de publicação?'

    # create dummy document
    dummy_document = {}
    dummy_document['text'] = document['text'] if 'text' in document.keys() else  document['texto']
    dummy_document['uuid'] = document['uuid']

    # exclude crazy chars
    dummy_document['text'] = dummy_document['text'].replace('༡༨/༢','')

    # extract the context(s) and respective offset(s)
    contexts, offsets = get_context(
        dummy_document,
        context_content=context_content,
        max_size=max_size,
        start_position=0,
        proportion_before=0.2,
        return_position_offset=True,
        use_sentence_id=use_sentence_id,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        question=largest_question,
        window_overlap=window_overlap,
        max_windows=max_windows)
    if not isinstance(contexts, list):
        contexts = [contexts]
        offsets = [offsets]

    # document structure in SQuAD format
    document_data = {
        "title": document['uuid'],
        "paragraphs": []
    }
    counter_qas = 0

    for context, _ in zip(contexts, offsets):
        # create one paragraph for each context.
        # it will be unique, except for windows-based context_contents
        paragraph = {
            "context": context,
            "qas": [],
        }
        paragraph_counter_qas = 0

        # control which of the requested qa_ids were satified. It force not-applicable
        # qas for qa_ids whose information does not exist in the dataset.
        all_qa_ids_satisfied = []

        # We will use only the fields listed in list_of_type_names
        for qa_id in list_of_type_names:
            doc_type = qa_id.split('.')[0]
            if doc_type != document_type:
                continue

            if qa_id in list_of_use_compound_question:
                questions = get_questions_for_chunk(qa_id=qa_id, is_compound=True)
                qas = get_compound_question_answers(
                    document,
                    questions=questions,
                    qa_id=qa_id,
                    choose_question=choose_question)
            else:
                questions = get_questions_for_chunk(qa_id=qa_id)
                qas = get_question_answers(document,
                                        questions=questions,
                                        qa_id=qa_id,
                                        choose_question=choose_question)
            
            paragraph_counter_qas += len(qas)

            # Include the question-answer of the current type_name (e.g., tipo)
            # in the current paragraph of the current document
            for qa in qas:
                paragraph["qas"].append(qa)
                all_qa_ids_satisfied.append(qa_id)

        # extract not-applicable qas for non-existent information.
        add_not_applicable = sorted(
            list(set(all_qa_ids) - set(all_qa_ids_satisfied))
        )

        for qa_id in add_not_applicable:

            qas = get_notapplicable_question_answers(
                qa_id=qa_id,
                choose_question='first',  # avoid using too much negatives
                list_of_use_compound_question=list_of_use_compound_question)

            paragraph_counter_qas += len(qas)

            # Include the not-applicable question-answer in the current
            # paragraph of the current document
            for qa in qas:
                paragraph["qas"].append(qa)
                all_qa_ids_satisfied.append(qa_id)

        # Add the current paragraph in the structure
        if paragraph_counter_qas > 0:
            document_data["paragraphs"].append(paragraph)
            counter_qas += paragraph_counter_qas

    return document_data, counter_qas
