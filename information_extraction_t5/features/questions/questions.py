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

_QUESTIONS_FORM = {
    'etiqueta':             [
        'Qual é o número da etiqueta?',
    ],
    'agencia':              [
        'Qual é o número da agência?',
    ],
    'conta_corrente':       [
        'Qual é o número da conta corrente?',
    ],
    'cpf':                  [
        'Qual é o CPF/CNPJ?',
        'Qual é o CPF do titular?',
    ],
    'nome_completo':        [
        'Qual é o nome?',
        'Qual é o nome completo?',
    ],
    'n_doc_serie':          [
        'Qual é o número do documento ou número da série?',
    ],
    'orgao_emissor':        [
        'Qual é o órgão emissor?',
    ],
    'doc_id_uf':            [
        'Qual é o estado do documento de identificação?',
        'Qual é a UF do documento de identificação?',
    ],
    'data_emissao':         [
        'Qual é a data de emissão?',
    ],
    'data_nascimento':      [
        'Qual é a data de nascimento?',
    ],
    'nome_mae':             [
        'Qual é o nome da mãe?',
    ],
    'nome_pai':             [
        'Qual é o nome do pai?',
    ],
    'endereco': OrderedDict({
        'compound':         [
            'Qual o endereço?',
        ],
        'logradouro':       [
            'Qual é o logradouro?',
        ],
        'numero':           [
            'Qual é o número?',
        ],
        'complemento':      [
            'Qual é o complemento?',
        ],
        'bairro':           [
            'Qual é o bairro?',
        ],
        'cidade':           [
            'Qual é a cidade?',
        ],
        'estado':           [
            'Qual é o estado?',
        ],
        'cep':              [
            'Qual é o CEP?',
        ]
    }),
}

# Include here other pairs (project, questions dict) for new datasets
QUESTIONS = {
    'form': _QUESTIONS_FORM,
}
