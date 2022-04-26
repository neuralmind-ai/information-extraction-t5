"""Dictionaries that map type-names to types and vice-versa. The types are
used as clues in brackets in T5 outputs. The type-names are recovered in
post-processing stage.

Each document class (project) has it own TYPENAME_TO_TYPE dictionary. We
strongly recommend that the types used in all the projects be consistent, and
as generic as possible. For example, using `CPF/CNPJ` for all CPFs and CNPJs,
regardless of being a consultant, current account holder, business partner,
land owner, etc.
"""
COMPLEMENT_TYPE = 'aparece no texto'  # or 'appears in the text' for EN

# Create a _NEWDATASET_TYPENAME_TO_TYPE for each new dataset, and
# update the TYPENAME_TO_TYPE dict.

_FORM_TYPENAME_TO_TYPE = {
    "etiqueta":                  "Etiqueta",
    "agencia":                   "Agência",
    "conta_corrente":            "Conta Corrente",
    "cpf":                       "CPF/CNPJ",
    "nome_completo":             "Nome",
    "n_doc_serie":               "No do Documento",
    "orgao_emissor":             "Órgão Emissor",
    "data_emissao":              "Data de Emissão",
    "data_nascimento":           "Data de Nascimento",
    "nome_mae":                  "Nome da Mãe",
    "nome_pai":                  "Nome do Pai",
    "endereco":                  "Endereço",
    "logradouro":                "Logradouro",
    "numero":                    "Número",
    "complemento":               "Complemento",
    "bairro":                    "Bairro",
    "cidade":                    "Cidade",
    "estado":                    "Estado",
    "cep":                       "CEP"
}

TYPENAME_TO_TYPE = {
    COMPLEMENT_TYPE: COMPLEMENT_TYPE,
}
TYPENAME_TO_TYPE.update(_FORM_TYPENAME_TO_TYPE)
# TYPENAME_TO_TYPE.update(_NEWDATASET_TYPENAME_TO_TYPE)

# This dict is used to recover the type-name by using the type. It is not
# critical to recover exactly the original type-name (different typenames
# can be mapped to the same type). Those type-names will be used in post
# processing after splitting sentences.
TYPE_TO_TYPENAME = {v: k for k, v in TYPENAME_TO_TYPE.items()}
