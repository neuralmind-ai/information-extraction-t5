"""Map of all questions, subquestions and its corresponding names.
Compound type-names must be represented as a OrderedDict with 'compound' and
subchunks' type-names as keys (even with empty lists) to keep a signature used
to prepare the compound answers.

A question name might have associated with it one of two things:
1. A list of questions.
2. A dictionary containing names of subquestions, which have their own list of
    subquestions.
"""
from collections import OrderedDict
from typing import Dict, List, Union

SUBQUESTION_NAME = str  # Ex.: 'rua'
SUBQUESTION = str  # Ex.: 'Qual a rua?'
QUESTION_NAME = str  # Ex.: 'endereco'
QUESTION = str  # Ex.: 'Qual o endereco?'
SUBQUESTION_DICT = Dict[SUBQUESTION_NAME, List[SUBQUESTION]]
QUESTION_DICT = Dict[QUESTION_NAME, Union[SUBQUESTION_DICT, List[QUESTION]]]

COMPLEMENT = ' e como aparece no texto?'  # or 'and how does it appear in the text?' for EN

_QUESTIONS_PUBLICACAO = {
    'instancia':            [
        'Qual é a instância?',
        'A qual instância pertence a publicação?',
        'Qual é a instância responsável pela publicação?',
    ],
    'orgao':                [
        'Qual é o órgão?',
        'A qual órgão pertence a publicação?',
        'Qual é a órgão associado à publicação?',
    ],
    'tipoPublicacao':       [
        'Qual é o tipo?',
        'A publicação é de qual tipo?',
        'Qual é o tipo de publicação do documento?',
    ],
    'instancia_orgao_tipo': OrderedDict({
        'compound':         [
            'Quais são as informações essenciais da publicação?',
            'Quais são a instância, o órgão e o tipo da publicação?',
            'Quais são as principais informações do documento de publicação?',
        ],
        'instancia':        [],
        'orgao':            [],
        'tipoPublicacao':   [],
        }),
}

# Include here other pairs (project, questions dict) for new datasets
QUESTIONS = {
    'publicacoes': _QUESTIONS_PUBLICACAO,
}
