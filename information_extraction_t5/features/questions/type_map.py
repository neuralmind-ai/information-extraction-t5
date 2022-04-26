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

_PUBLICACAO_TYPENAME_TO_TYPE = {
    "instancia":                    "Instância",
    "orgao":                        "Órgão",
    "tipoPublicacao":               "Tipo"
}

TYPENAME_TO_TYPE = {
    COMPLEMENT_TYPE: COMPLEMENT_TYPE,
}
TYPENAME_TO_TYPE.update(_PUBLICACAO_TYPENAME_TO_TYPE)
# TYPENAME_TO_TYPE.update(_NEWDATASET_TYPENAME_TO_TYPE)

# This dict is used to recover the type-name by using the type. It is not
# critical to recover exactly the original type-name (different typenames
# can be mapped to the same type). Those type-names will be used in post
# processing after splitting sentences.
TYPE_TO_TYPENAME = {v: k for k, v in TYPENAME_TO_TYPE.items()}
