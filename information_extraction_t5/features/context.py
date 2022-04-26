import math
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokens_and_offsets(text: str, tokenizer: PreTrainedTokenizerBase) -> List[Tuple[Any, int, int]]:
    tokens = tokenizer.tokenize(text)
    token_lens = [len(token) for token in tokens]
    token_lens[0] -= 1  # Ignore first "_" token
    token_ends = np.cumsum(token_lens)
    token_starts = [0] + token_ends[:-1].tolist()
    tokens_and_offsets = list(zip(tokens, token_starts, token_ends))
    return tokens_and_offsets


def get_token_id_from_position(tokens_and_offsets: List[Tuple[Any, int, int]], position: int) -> int:
    for idx, tok_offs in enumerate(tokens_and_offsets):
        _, start, end = tok_offs
        if start <= position < end:
            return idx
    return len(tokens_and_offsets) - 1


def get_max_size_context(document: Dict, max_size: int = 4000, question: str = 'Qual?') -> str:
    """Returns the first max_size characters of the document_text.
    """
    document_text = document['text']
    question_sentence = f'question: {question} context: '
    num_chars_question = len(question_sentence)
    remaining_chars = max_size - num_chars_question

    context = document_text[:remaining_chars - 4]
    context = context + ' ...'
    return context


def get_position_context(
    document: Dict,
    max_size: int = 4000,
    start_position: int = 0,
    proportion_before: float = 0.2,
    question: str = 'Qual?',
    use_sentence_id: bool = False,
    verbose: bool = False,
    ) -> Tuple[str, int]:
    """Returns the content around a specific position with size controlled by max_size.
    proportion_before indicates the proportion of max_size the must be taken before 
    the position, while 1 - position_before is after.
    """
    document_text = document['text']
    question_sentence = f'question: {question} context: '
    num_chars_question = len(question_sentence)

    remaining_chars = max_size - num_chars_question
    start_reticences, end_reticences = False, False

    start = math.floor(remaining_chars * proportion_before)
    start = max(0, start_position - start)
    end = min(len(document_text), remaining_chars + start)

    if use_sentence_id:
        num_chars_each_sentence_id = len('[SENT1]')
        num_chars_sentence_id = (document_text[start: end].count('\n') + 1) * num_chars_each_sentence_id
    else:
        num_chars_sentence_id = 0
    size = end - start

    # remove chars if current size + sentence-ids chars exceed the remaining chars
    if size + num_chars_sentence_id > remaining_chars:
        to_remove = (size + num_chars_sentence_id) - remaining_chars
        
        # Chars are removed fractionally (20 times), in order to control the 
        # expected size, avoiding exaggerated removal. In each iteration, as 
        # long as the the window size is updated, the num_chars_sentence_id
        # is updated as well.
        to_remove_fractions = [to_remove // 20] * 20 + [to_remove % 20]

        for to_remove in to_remove_fractions:
            if start == start_position:
                end -= to_remove
            else:
                remove_before = math.floor(to_remove * proportion_before)
                remove_before = min(remove_before, start_position - start)
                remove_after = to_remove - remove_before
                start += remove_before
                end -= remove_after 

            num_chars_sentence_id = (document_text[start: end].count('\n') + 1) * num_chars_each_sentence_id 
            size = end - start

            # the size satifies remaining_tokens
            if size + num_chars_sentence_id <= remaining_chars:
                break

    # check if it requires reticences
    # if it does, try to find a space before/after the start_position
    if start != 0:
        start_reticences = True
        start = max(start, document_text.find(' ', start, start_position))
        position_offset = start - 3  # reticences
    else:
        position_offset = start
    
    if end < len(document_text):
       end_reticences = True
       end = document_text.rfind(' ', start_position, end)

    if verbose:
        print('-- MUST CONTAIN: ' + document_text[start_position: start_position+30])
        print(f'-- start: {start}, end: {end}')
        c = document_text[start:end]
        print(f'-- len (char): {len(c)}')
        print(f'-- context: {c} \n')
    
    context = ('...' if start_reticences else '') \
        + document_text[start: end] \
        + ('...' if end_reticences else '')

    if verbose:
        # it can exceed the expected num of chars because of reticences.
        print('--> testing the number of chars:')
        t5_input = question_sentence + context
        n = len(t5_input)
        print(f'>> The input occupies {n} chars. '
            f'It will have additional {num_chars_sentence_id} for sentence-ids. '
            f'Total: {n + num_chars_sentence_id}. Expected: {max_size}.')
    
    return context, position_offset


def get_windows_context(
    document: Dict,
    max_size: int = 4000,
    window_overlap: float = 0.5,
    max_windows: int = 3,
    question: str = 'Qual?',
    use_sentence_id: bool = False,
    verbose: bool = False,
    ) -> Tuple[List[str], List[int]]:
    """Returns a list of window contents with size controlled by max_size, with
    overlapping near to 50%.
    """
    document_text = document['text']

    assert max_windows != 0, (
        'Set max_windows higher than 0 to get a specific quantity of windows, '
        'or below to extract all possible ones.')

    contexts, offsets = [], []

    start_position, position_offset = 0, 0
    context = ''
    # the offset + current context size surpassing document size means the 
    # window reached the end of document
    while position_offset + len(context) < len(document_text):

        context, position_offset = get_position_context(document, max_size=max_size,
            start_position=start_position, proportion_before=0, question=question, 
            use_sentence_id=use_sentence_id, verbose=verbose)

        contexts.append(context)
        offsets.append(position_offset)

        if verbose:
            print(f'>>>>>>>>>> WINDOW: start_position = {start_position}, offset = {position_offset}')

        start_position += int(len(context) * (1 - window_overlap))

        if max_windows > 0 and len(contexts) == max_windows: break

    return contexts, offsets


def get_token_context(document: Dict, 
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    question: str = 'Qual?',
    use_sentence_id: bool = False,
    verbose: bool = False,
    ) -> Tuple[str, int]:
    """Returns the first max_tokens tokens of the document_text.
    """
    context, position_offset = get_position_token_context(document, start_position=0,
        proportion_before=0, tokenizer=tokenizer, max_tokens=max_tokens, question=question,
        use_sentence_id=use_sentence_id, verbose=verbose)
    return context, position_offset


def get_position_token_context(
    document: Dict,
    start_position: int = 0,
    proportion_before: float = 0.2,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    tokens_and_offsets: Optional[List[Tuple[Any, int, int]]] = None,
    question: str = 'Qual?',
    use_sentence_id: bool = False,
    verbose: bool = False,
    ) -> Tuple[str, int]:
    """Returns the content around a specific position, with size controlled by max_tokens.
    proportion_before indicates the proportion of max_size the must be taken before the 
    position, while 1 - position_before is after.
    """
    document_text = document['text']
    question_sentence = f'question: {question} context: '
    num_tokens_question = len(tokenizer.tokenize(question_sentence))

    remaining_tokens = max_tokens - num_tokens_question
    start_reticences, end_reticences = False, False

    if tokens_and_offsets is None:
        tokens_and_offsets = get_tokens_and_offsets(text=document_text, tokenizer=tokenizer)
    positional_token_id = get_token_id_from_position(tokens_and_offsets=tokens_and_offsets, position=start_position)
    start_token_id = max(0, positional_token_id - math.floor(remaining_tokens * proportion_before))
    end_token_id = min(positional_token_id + math.ceil(remaining_tokens * (1-proportion_before)), len(tokens_and_offsets))

    start = tokens_and_offsets[start_token_id][1]
    end = tokens_and_offsets[end_token_id-1][2]

    if use_sentence_id:
        num_tokens_each_sentence_id = len(tokenizer.tokenize('[SENT10]'))
        num_tokens_sentence_id = (document_text[start: end].count('\n') + 1) * num_tokens_each_sentence_id 
    else:
        num_tokens_sentence_id = 0
    size = end_token_id - start_token_id

    # remove tokens if current size + sentence-ids tokens exceed the remaining tokens
    if size + num_tokens_sentence_id > remaining_tokens:
        to_remove = (size + num_tokens_sentence_id) - remaining_tokens
        
        # Tokens are removed fractionally (20 times), in order to control the 
        # expected size, avoiding exaggerated removal. In each iteration, as 
        # long as the the window size is updated, the num_tokens_sentence_id
        # is updated as well.
        to_remove_fractions = [to_remove // 20] * 20 + [to_remove % 20]

        for to_remove in to_remove_fractions:
            if start == start_position:
                end_token_id -= to_remove
            else:
                remove_before = math.floor(to_remove * proportion_before)
                remove_before = min(remove_before, positional_token_id - start_token_id)
                remove_after = to_remove - remove_before
                start_token_id += remove_before
                end_token_id -= remove_after 

            start = tokens_and_offsets[start_token_id][1]
            end = tokens_and_offsets[end_token_id-1][2]

            num_tokens_sentence_id = (document_text[start: end].count('\n') + 1) * num_tokens_each_sentence_id 
            size = end_token_id - start_token_id

            # the size satifies remaining_tokens
            if size + num_tokens_sentence_id <= remaining_tokens:
                break
        
    # check if it requires reticences
    # if it does, try to find a space before/after the start_position
    if start != 0:
        start_reticences = True
        start = max(start, document_text.find(' ', start, start_position))
        position_offset = start - 3  # reticences
    else:
        position_offset = tokens_and_offsets[start_token_id][1]

    if end < len(document_text):
       end_reticences = True
       end = document_text.rfind(' ', start_position, end)

    if verbose:
        print('-- MUST CONTAIN: ' + document_text[start_position: start_position+30])
        print(f'-- start: {start}, end: {end}')
        c = document_text[start: end]
        print(f'-- len (char): {len(c)}')
        print(f'-- len (toks): {end_token_id - start_token_id}')
        print(f'-- context: {c} \n')

    context = ('...' if start_reticences else '') \
        + document_text[start: end] \
        + ('...' if end_reticences else '')

    if verbose:
        # it can exceed the expected num of tokens because of reticences.
        print('--> testing the number of tokens:')
        t5_input = question_sentence + context
        n = len(tokenizer.tokenize(t5_input))
        print(f'>> The input occupies {n} tokens. '
            f'It will have additional {num_tokens_sentence_id} for sentence-ids. '
            f'Total: {n + num_tokens_sentence_id}. Expected: {max_tokens}.')

    return context, position_offset 


def get_windows_token_context(
    document: Dict,
    window_overlap: float = 0.5,
    max_windows: int = 3,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    question: str = 'Qual?',
    use_sentence_id: bool = False,
    verbose: bool = False,
    ) -> Tuple[List[str], List[int]]:
    """Returns a list of window contents with size controlled by max_tokens, with
    overlapping near to 50%.
    """
    document_text = document['text']

    assert max_windows != 0, (
        'Set max_windows higher than 0 to get a specific quantity of windows, '
        'or below to extract all possible ones.')
    
    contexts, offsets = [], []
    tokens_and_offsets = get_tokens_and_offsets(text=document_text, tokenizer=tokenizer)

    assert len(document_text) == tokens_and_offsets[-1][2], (
        f'The original document ({document["uuid"]}) and the end of last token are not matching: {len(document_text)} != {tokens_and_offsets[-1][2]}')

    start_position, position_offset = 0, 0
    context = ''
    # the offset + current context size surpassing document size means the 
    # window reached the end of document
    while position_offset + len(context) < len(document_text):

        context, position_offset = get_position_token_context(document, start_position=start_position,
            proportion_before=0, tokenizer=tokenizer, max_tokens=max_tokens, tokens_and_offsets=tokens_and_offsets, 
            question=question, use_sentence_id=use_sentence_id, verbose=verbose)

        contexts.append(context)
        offsets.append(position_offset)

        if verbose:
            print(f'>>>>>>>>>> WINDOW: start_position = {start_position}, offset = {position_offset}')

        start_position += int(len(context) * (1 - window_overlap))

        if max_windows > 0 and len(contexts) == max_windows: break

    return contexts, offsets


def get_context(
    document: Dict,
    context_content: str = 'windows_token',
    max_size: int = 4000,
    start_position: int = 0,
    proportion_before: float = 0.2,
    return_position_offset: bool = False,
    use_sentence_id: bool = False,
    tokenizer: Union[None, PreTrainedTokenizerBase] = None,
    max_tokens: int = 512,
    question: str = 'Qual?',
    window_overlap: float = 0.5,
    max_windows: int = 3,
    verbose: bool = False,
    ) -> Union[str, List[str], Tuple[Union[str, List[str]], Union[int, List[int]]]]: 
    """Returns the context to use in T5 input based on context_content.
        
     Args:
        document: dict with all the information of current document.
        context_content: type of context (max_size, position, token, 
            position_token or windows_token).
            - max_size: gets the first max_size characters.
            - position: gets a window text limited to max_size characters 
            around a start_position, respecting a proportion before and after 
            the position.
            - windows: gets a list of sliding windows of max_size, comprising
            the complete document.
            - token: gets the first max_tokens tokens.
            - position_token: gets a window text limited to max_tokens tokens
            around a start_position, respecting a proportion before and after 
            the position, and penalizing tokens that will be occupied by 
            question and sentence-ids in the T5 input.
            - windows_token: gets a list of sliding windows of max_tokens,
            comprising the complete document.
        max_size: maximum size of context, in chars (used for max_size and
            position).
        start_position: char index of a keyword in the original document text 
            (used for position and position_token).
        proportion_before: proportion of maximum context size (max_size or 
            max_tokens) that must be before start_position (used for position, 
            position_token and the variants).
        return_position_offset: if True, returns the position of returned 
            context with respect to original document text (used for position, 
            position_token and the variants).
        tokenizer: AutoTokenizer used in the model (used for position_token and 
            windows_token).
        max_tokens: maximum size of context, in tokens (used for position_token 
            and windows token).
        question: question that will be used along with the context in the T5
            input (used for position_token and windows_token).
        window_overlap: overlapping between windows (used for windows and 
            windows_token).
        max_windows: the maximum number of windows to generate, use -1 to get
            all the possible windows (used for windows and windows_token)
        verbose: visualize the processing, tests, and resultant contexts.

    Returns:
        - the context.
        - the position_offset (optional).        
    """
    position_offset = 0

    # remove repeated breaklines, repeated spaces/tabs, space/tabs before
    # breaklines, and breaklines in start/end of document text to make the token 
    # positions match the char positions. Those rules avoid incorrect alignments.
    document['text'] = document['text'].replace('\t', ' ')            # '\t'
    document['text'] = re.sub(r'\s*\n+\s*', r'\n', document['text'])  #  space (0 or more) + '\n' (1 or more) + space (0 or more)
    document['text'] = re.sub(r'(\s)\1+', r'\1', document['text'])    # space (1 or more)
    # special characters that causes raw and tokinization texts to desagree
    document['text'] = document['text'].replace('´', '\'')          # 0 char --> 1 char in tokenization (common in publicacoes)
    document['text'] = document['text'].replace('™', 'TM')          # 1 char --> 2 chars in tokenization
    document['text'] = document['text'].replace('…', '...')         # 1 char --> 3 chars in tokenization
    document['text'] = document['text'].strip()

    if context_content == 'max_size':
        context = get_max_size_context(document, max_size=max_size, question=question)
    elif context_content == 'position':
        context, position_offset = get_position_context(document, max_size=max_size, 
            start_position=start_position, proportion_before=proportion_before, 
            question=question, use_sentence_id=use_sentence_id, verbose=verbose)
    elif context_content == 'windows':
        context, position_offset = get_windows_context(document, max_size=max_size,
            window_overlap=window_overlap, max_windows=max_windows,
            question=question, use_sentence_id=use_sentence_id, verbose=verbose)
    elif context_content == 'token':
        context, position_offset = get_token_context(document, 
            tokenizer=tokenizer, max_tokens=max_tokens, question=question,
            use_sentence_id=use_sentence_id, verbose=verbose)
    elif context_content == 'position_token':
        context, position_offset = get_position_token_context(document, start_position=start_position,
            proportion_before=proportion_before, tokenizer=tokenizer, max_tokens=max_tokens,
            question=question, use_sentence_id=use_sentence_id, verbose=verbose)
    elif context_content == 'windows_token':
        context, position_offset = get_windows_token_context(document,
            window_overlap=window_overlap, max_windows=max_windows, tokenizer=tokenizer,
            max_tokens=max_tokens, question=question, use_sentence_id=use_sentence_id, verbose=verbose)
    else:
        return '', position_offset

    if verbose:
        if isinstance(context, list):
            for (i, cont) in enumerate(context):
                print(f'--------\nWINDOW {i}\n--------')
                print(f'len: {len(cont)} context: {cont} \n')
        else:
            print(f'len: {len(context)} context: {context} \n')

    if return_position_offset:
        return context, position_offset
    else:
        return context


def main():
    document = {}
    document['uuid'] = '1234567'
    document['text'] = "Que tal fazer uma poc inicial para vermos a viabilidade e identificarmos as dificuldades?\nA motivação da escolha desse problema " \
    "foi que boa parte dos atos de matrícula passam de 512 tokens, e ainda não temos uma solução definida para fazer treinamento e predições em " \
    "janelas usando o QA.\nEssa limitação dificulta o uso de QA para problemas que não sabemos onde a informação está no documento (por enquanto, " \
    "só aplicamos QA em tarefas que sabemos que a resposta está nos primeiros 512 tokens da matrícula).\nComo esse problema de identificar a proporção " \
    "de cada pessoa são duas tarefas (identificação + relação com uma pessoa), podemos usar a localização da pessoa no texto para selecionar apenas " \
    "uma pedaço do ato de alienação pra passar como contexto pro modelo, evitando um pouco essa limitação dos 512 tokens."
    document['text'] = "PREFEITURA DE CAUCAIA\nSECRETARIA DE FINAN\u00c7AS,PLANEJAMENTO E OR\u00c7AMENTO\nCERTID\u00c3O NEGATIVA DE TRIBUTOS ECON\u00d4MICOS\nLA SULATE\nN\u00ba 2020000982\nRaz\u00e3o Social\nCOMPASS MINERALS AMERICA DO SUL INDUSTRIA E COMERC\nINSCRI\u00c7\u00c3O ECON\u00d4MICA Documento\nBairro\n00002048159\nC.N.P.J.: 60398138001860\nSITIO SALGADO\nLocalizado ROD CE 422 KM 17, S/N - SALA SUPERIOR 01 CXP - CAUCAIA-CE\nCEP\n61600970\nDADOS DO CONTRIBUINTE OU RESPONS\u00c1VEL\nInscri\u00e7\u00e3o Contribuinte / Nome\n169907 - COMPASS MINERALS AMERICA DO SUL INDUSTRIA E COMERC\nEndere\u00e7o\nROD CE 422 KM 17, S/N SALA SUPERIOR 01 CXP\nDocumento\nC.N.P.J.: 60.398.138/0018-60\nSITIO SALGADO CAUCAIA-CE CEP: 61600970\nNo. Requerimento\n2020000982/2020\nNatureza jur\u00eddica\nPessoa Juridica\nCERTID\u00c3O\nCertificamos para os devidos fins, que revendo os registros dos cadastros da d\u00edvida ativa e de\ninadimplentes desta Secretaria, constata-se - at\u00e9 a presente data \u2013 n\u00e3o existirem em nome do (a)\nrequerente, nenhuma pend\u00eancia relativa a tributos municipais.\nSECRETARIA DE FINAN\u00c7AS, PLANEJAMENTO E OR\u00c7AMENTO se reserva o direito de inscrever e cobrar as\nd\u00edvidas que posteriormente venham a ser apurados. Para Constar, foi lavrada a presente Certid\u00e3o.\nA aceita\u00e7\u00e3o desta certid\u00e3o est\u00e1 condicionada a verifica\u00e7\u00e3o de sua autenticidade na internet, nos\nseguinte endere\u00e7o: http://sefin.caucaia.ce.gov.br/\nCAUCAIA-CE, 03 DE AGOSTO DE 2020\nEsta certid\u00e3o \u00e9 v\u00e1lida por 090 dias contados da data de emiss\u00e3o\nVALIDA AT\u00c9: 31/10/2020\nCOD. VALIDA\u00c7\u00c3O 2020000982"
    # document['text'] = "DESAANZ\nJUCESP - Junta Comercial do Estado de S\u00e3o Paulo\nMinist\u00e9rio do Desenvolvimento, Ind\u00fastria e Com\u00e9rcio Exterios\nSamas\nSECRETAR\u00cdA DE DESENVOLVIMENTO\ndo Com\u00e9rcio - DNRC\nECONOMICO, CI\u00caNCIA,\nn\u00f4mico, Ci\u00eancia e Tecnologia\nTECNOLOGIA E INOVA\u00c7\u00c3O\nBestellen\nCERTIFICO O REGISTROFLAVIA REAT BRITTO\nSOB O N\u00daMERO SECRETARIA IGERAL EM EXERC\nda Reguerimento:\n5461/15-7 A LEHET BEDS\nSEQ. DOC.\n15 JAN. 2015\n1\nJUCESP\nHU\nSIP\nJUCESP PROTOCOLO\n0.024.119/15-5\n1\nJunta Comba\nEstado de S\u00e3o Paulo\n14\nJUNTA CON\nNubia Cristina da Silva Cembull\nAssessora T\u00e9cnica do Registro Publico\nR.G.: 36.431.427-3\nDADOS CADASTRAIS\n13 Hd\nCODIGO DE BARRAS (NIRE)\nCNPJ DA SEDE\nNIRE DA SEDE\n3522550861-2\n13.896.623/0001-36\nSEM EXIG\u00caNCIA ANTERIOR\nPROIE\nATO(S)\nAltera\u00e7\u00e3o de Endere\u00e7o; Altera\u00e7\u00e3o de Nome Empresarial; Consolida\u00e7\u00e3o da\nNOME EMPRESARIAL\nRF MOTOR'S Com\u00e9rcio de ve\u00edculos Ltda. - ME\n!\nLOGRADOURO\nAvenida Regente Feij\u00f3\nN\u00daMERO\n277\n:\nCEP\nCOMPLEMENTO\nBAIRRO/DISTRITO\nVila Regente Feij\u00f3\nC\u00d3DIGO DO MUNICIPIO\n5433\n03342-000\nUF\nMUNICIPIO\nS\u00e3o Paulo\nSP\nTELEFONE\nCORREIO ELETR\u00d4NICO\nIN, OAB\nU.F.\nNOME DO ADVOGADO\nVALORES RECOLHIDOS IDENTIFICA\u00c7\u00c3O DO REPRESENTANTE DA EMPRESA\nDARE 54,00\nNOME:\nBruno Vinicius Ferreira (S\u00f3cio )\nDARF 21,00\nASSINATURA:\nDATA ASSINATURA:\n12/01/2015\nB\nDECLARO, SOB AS PENAS DA LEI, QUE AS INFORMA\u00c7\u00d5ES CONSTANTES DO REQUERIMENTO/PROCESSO S\u00c3O EXPRESS\u00c3O DA VERDADE.\nControle Internet\n\u0421.\n015755122-9\n12/1/2015 10:19:14 - P\u00e1gina 1 de 2\n\n\n1ERCIAL\npy\nOLO\nINSTRUMENTO PARTICULAR DE ALTERA\u00c7\u00c3O\nCONTRATUAL DE SOCIEDADE EMPRES\u00c1RIA DE FORMA\nLIMITADA:\nREAVEL FERREIRA COM\u00c9RCIO DE VE\u00cdCULOS LTDA. ME\nCNPJ 13.896.623/0001-36\nPelo presente instrumento particular de altera\u00e7\u00e3o\ndo contrato social, os abaixo qualificados e ao final assinados:\nBruno Vinicius Ferreira, brasileiro, solteiro, nascido em 26/10/1985,\nempres\u00e1rio, portador da c\u00e9dula de identidade RG sob n\u00ba. 42.318.703-X/SSP-SP, inscrito no CPF/MF sob n\u00ba. 340.446.998-44, residente e\ndomiciliado no Estado de S\u00e3o Paulo, \u00e0 Rua Altina Penna Botto, 16 -\nCasa 02 - Vila Ivone - CEP 03375-001;\nDiogo Gabriel Ferreira, brasileiro, solteiro, nascido em 08/10/1988,\nempres\u00e1rio, portador da c\u00e9dula de identidade RG sob n\u00ba. 44.476.866-X/SSP-SP, inscrito no CPF/MF sob n\u00ba. 359.085.288-70, residente e\ndomiciliado no Estado de S\u00e3o Paulo, \u00e0 Rua Altina Penna Botto, 16 -\nCasa 02 - Vila Ivone - CEP 03375-001;\n\u00danicos s\u00f3cios da sociedade empres\u00e1ria de forma limitada que gira\na denomina\u00e7\u00e3o social de REAVEL FERREIRA\nCom\u00e9rcio de Ve\u00edculos Ltda. ME, inscrita no CNPJ/MF sob n\u00ba.\n13.896.623/0001-36, com estabelecimento e sede \u00e0 Rua Acuru\u00ed, 508 -\nVila Formosa S\u00e3o Paulo CEP 03355-000 S.P., cujos atos\nconstitutivos encontram-se registrados e arquivados na Junta\nComercial do Estado de S\u00e3o Paulo, com NIRE sob n\u00ba 35.2.25508612,\nem sess\u00e3o de 22 de Junho de 2011, t\u00eam, entre si justos e contratados\npromovem a altera\u00e7\u00e3o contratual e consequente consolida\u00e7\u00e3o da\nempresa que obedecera as clausulas e condi\u00e7\u00f5es adiante descritas:\nB\n\n\nVistoContenido\nRG: 36.430.427-3\nAltera\u00e7\u00e3o Contratual\nCl\u00e1usula 1a:- Altera-se a raz\u00e3o social da empresa que passa a ser\nRF MOTOR'S Com\u00e9rcio de Ve\u00edculos Ltda. - ME com denomina\u00e7\u00e3o\nde fantasia RF MOTOR'S;\nCl\u00e1usula 2a:- Altera-se o endere\u00e7o da sociedade que passa a ser \u00e0\nAv. Regente Feij\u00f3, 277 - Vila Regente Feij\u00f3 - S\u00e3o Paulo CEP\n03342-000 - S.P.;\nCl\u00e1usula 3a:- Face \u00e0s altera\u00e7\u00f5es\nos s\u00f3cios deliberam a\nCONSOLIDA\u00c7\u00c3O CONTRATUAL, conforme segue:\nCONTRATUAL\nCl\u00e1usula 1:- A sociedade girar\u00e1 sob a denomina\u00e7\u00e3o social de RF\nMOTOR'S Com\u00e9rcio Veiculos Ltda. - ME com denomina\u00e7\u00e3o de\nfantasia RF MOTOR'S, e ter\u00e1 a sua sede \u00e0 Av. Regente Feij\u00f3, 277 -\nVila Regente Feij\u00f3 - S\u00e3o Paulo - CEP 03342-000 - S.P.;\nCl\u00e1usula 2: sociedade tem por fim e objetivo na forma da\nlegisla\u00e7\u00e3o\nCom\u00e9rcio a varejo de autom\u00f3veis, camionetas e utilit\u00e1rios novos;\nCom\u00e9rcio por atacado de autom\u00f3veis, camionetas e utilit\u00e1rios\nnovos e usados;\nCom\u00e9rcio a varejo de autom\u00f3veis, camionetas e utilit\u00e1rios usados;\nCom\u00e9rcio por atacado de motocicletas e motonetas;\nCom\u00e9rcio a varejo de motocicletas e motone novas;\nCl\u00e1usula 3:- A sociedade teve in\u00edcio em 22 de Junho de 2011 e ter\u00e1\ndura\u00e7\u00e3o por tempo indeterminado;\n8.\n\n\n300 m\nVi\u015fte\nCl\u00e1usula 4:- 0 capital social \u00e9 de R$ 10.000,00 (Dez Mil Reais)\ntotalmente subscrito e integralizado em moeda corrente nacional,\nrepresentado por 10.000 (dez mil) cotas no valor unit\u00e1rio de R$ 1,00\n(Hum Real) cada, assim distribu\u00eddo:-1. Bruno Vinicius Ferreira, 9.900 (nove mil e novecentas) cotas de\nvalor unit\u00e1rio de R$ 1,00 (Hum Real), totalizando R$ 9.900,00 (Nove\nMil e Novecentos Reais), totalmente subscritas e integralizadas em\nmoeda corrente nacional, neste ato;\n2. Diogo Gabriel Ferreira, 100 cotas de valor unit\u00e1rio de R$\n1,00 (Hum Real), totalizando R$ 100,00 (Cem Reais), totalmente\nsubscritas e integralizadas em moeda corrente nacional, neste ato;\nCl\u00e1usula 5:- A responsabilidade dos s\u00f3cios \u00e9 restrita ao valor de suas\ncotas, mas todos\ndo Capital\npela integraliza\u00e7\u00e3o\ndeliberam que a administra\u00e7\u00e3o da sociedade,\nbem como sua representa\u00e7\u00e3o ativa e passiva, judicial ou extrajudicial,\nser\u00e1 exercida pelo s\u00f3cio Bruno Vinicius Ferreira individual e\nisoladamente. Inclusive todos os documentos legais e banc\u00e1rios, que\npoder\u00e1 constituir procuradores com tais poderes.\nPar\u00e1grafo Primeiro:- Os s\u00f3cios ter\u00e3o direito, a uma retirada mensal\na t\u00edtulo de Pr\u00f3-Labore e poder\u00e3o efetuar a distribui\u00e7\u00e3o de lucro, desde\nque, fixado em comum acordo no in\u00edcio de cada exerc\u00edcio.\nPar\u00e1grafo Segundo:- Os s\u00f3cios far\u00e3o uso da firma, podendo assinar\nseparadamente, ficando-lhes vedado, entretanto, o uso da firma em\nneg\u00f3cios alheios aos do objetivo social; e t\u00edtulos de responsabilidade\nsocial de esp\u00e9cie alguma, tais como avais, endossos, fian\u00e7as, etc.\n\n\nVisto\nConitor\n.RG36.430.427-3\nPar\u00e1grafo Terceiro:- A onera\u00e7\u00e3o ou venda de bens im\u00f3veis depende\nda expressa anu\u00eancia de s\u00f3cios que representem pelo menos 75%\n(setenta e cinco por cento) das quotas com direito a voto, respondendo\nos administradores solidariamente perante a sociedade e os terceiros\nprejudicados, por culpa no desempenho de suas fun\u00e7\u00f5es , de acordo\ncom o disposto no art. 1016, da Lei n. 10.406 de 10 de janeiro de\n2002.\nPar\u00e1grafo Quarto:- Depender\u00e1 tamb\u00e9m de expressa anu\u00eancia dos\ns\u00f3cios, conforme o disposto da Lei n. 10.406 de 10 de janeiro de\n2002, ficando assim solidariamente respons\u00e1vel civil e criminalmente\no s\u00f3cio que infringir o presente artigo:-a) Alienar, onerar ou de qualquer forma dispor de t\u00edtulos imobili\u00e1rios,\nbem como cotas ou a\u00e7\u00f5es de que a sociedade seja titular no capital de\noutras empresas;\nb) Fixar remunera\u00e7\u00e3o dos adminis\nistradores e assessores, sem v\u00ednculo\nempregat\u00edcio, a eles subordinados.\nCl\u00e1usula 7 :- Faculta-se a qualquer dos s\u00f3cios, retirar-se da sociedade\ndesde que o fa\u00e7a mediante aviso pr\u00e9vio de sua resolu\u00e7\u00e3o ao outro\ns\u00f3cio, observado o direito de prefer\u00eancia, com anteced\u00eancia m\u00ednima\nde pelo menos 6 (Seis) meses. Seus haveres lhes ser\u00e3o pagos em 12\n(Doze) meses corrigidos pelo IGPM e o primeiro vencimento \u00e0 partir\nde 60 (sessenta) dias da data do Balan\u00e7o Especial.\nCl\u00e1usula 8:- Os lucros e perdas apurados regularmente em balan\u00e7o\nanual que se realizar\u00e1 no dia 31 de Dezembro de cada ano, ser\u00e3o\ndivididos proporcionalmente ao capital social de cada um dos s\u00f3cios,\nem eventual preju\u00edzo os s\u00f3cios poder\u00e3o optar pelo aumento de capital\npara saldar tais preju\u00edzos.\n\n\nVisto\n15\nConfezidb\nRG: 15.530427-3\nCl\u00e1usula 9:- Em caso de falecimento de qualquer um dos s\u00f3cios, na\nvig\u00eancia do presente contrato, n\u00e3o importa na extin\u00e7\u00e3o da sociedade e\nseus neg\u00f3cios, cabendo ao s\u00f3cio remanescente a apura\u00e7\u00e3o dos haveres\ndo s\u00f3cio ausente segundo balan\u00e7o especial na data do \u00f3bito e, ser\u00e3o\npagos aos herdeiros do falecido em 12 (Doze) presta\u00e7\u00f5es mensais\ncorrigidos pelo IGPM, sendo vedado aos herdeiros poss\u00edvel ingresso\nna sociedade.\nCl\u00e1usula 10\u00b0:- Nenhum dos s\u00f3cios, pessoalmente ou por interposta\npessoa, poder\u00e1 participar ou colaborar a qualquer t\u00edtulo em outra\npessoa jur\u00eddica, que tenha por qualquer forma atividade an\u00e1loga ou\nconcorrente \u00e0 da sociedade, sem expressa anu\u00eancia dos demais.\nCl\u00e1usula 11:- Os administradores declaram, sob as penas da Lei, de\nque n\u00e3o est\u00e3o impedidos de exercerem a administra\u00e7\u00e3o da sociedade,\npor lei especial, ou em virtude de condena\u00e7\u00e3o criminal, ou por se\nencontrarem sob os efeitos dela, a pena que vede, ainda que\ntemporariamente, o acesso a cargos p\u00fablicos; ou por crime falimentar,\nde prevarica\u00e7\u00e3o, peita ou subomo, concuss\u00e3o, peculato, ou contra a\neconomia popular, contra o sistema financeiro nacional, contra\nnormas de defesa da concorr\u00eancia, contra as rela\u00e7\u00f5es de consumo, f\u00e9\np\u00fablica, ou a\n. (art. 1.011, \u00a71\u00baCC/2002).\nCl\u00e1usula 12 :- Para os casos omissos neste contrato, os mesmos ser\u00e3o\nregidos pelas disposi\u00e7\u00f5es legais vigentes atinentes \u00e0 mat\u00e9ria, em\nespecial a Lei n. 10.406, de 10 de janeiro de 2002.\nCl\u00e1usula 134:- Os s\u00f3cios elegem o foro Central da Comarca da\nCapital, no Estado de S\u00e3o Paulo, para as eventuais quest\u00f5es que\npossam advir.\nB\n\n\n..\n..\nVisto\nConletico\nE, assim, por estarem em tudo justos e contratados, as partes\nassinam o presente instrumento em 03 (tr\u00eas) vias de igual teor e valor\npara um s\u00f3 efeito, tudo, ante duas testemunhas a tudo presentes que\ntamb\u00e9m assinam, devendo em seguida ser encaminhado para registro\ne arquivamento junto a JU ESP - Junta Comercial do Estado de S\u00e3o\nPaulo.\nS\u00e3o Paulo, 12 de Janeiro de 2015.\nGolul tenuina\nBruno Vinicius Ferreira\nDiogo Gabriel Ferreira\nTestemunhas:\nRicardo\nHellon Austina da s Santos\nRicardo Silva Bezerra\nRG n\u00ba 29.074.987-6/SSP-SP\nCPF n\u00ba 213.108.838-82\nHellen Cristina da Silva Santos\nRG n\u00b037965378-3/SSP-SP\nCPF n\u00ba 405.216.528-47\nDO\n15 JAN. 2015\nwww.\nSASA\nSECRETARIA DE DESENVOLVIMENTO\n01/ECON\u00d3MICO, CI\u00caNCIA,\nTECNOLOGIA E INOVA\u00c7\u00c3O\nom\nCERTIFICO O REGISTRO FLAVTA REOTTA eri to\nSOB O NUMERO SECRET\u00c1RIA GERAL EM EXERCICIO\n5.461/15-7 tena MRITH FUIT\n...www.si\n\n\nDocumento B\u00e1sico de Entrada\nPage 1 of 1\n...\nREP\u00daBLICA FERERATIVA DO BRASIL\nCADASTRO NACIONAL.JA PESSOA JUR\u00cdDICA - CNPJ\nPROTOCOLO DE TRANSMISS\u00c3O DA FCP JUOVI30\nA an\u00e1lise e o deferimento deste documento ser\u00e3o efetuados pelo seguinte \u00f3rg\u00e3o:\n\u2022 Junta Comercial do Estado de S\u00e3o Paulo\nC\u00d3DIGO DE ACESSO\nSP.63.05.42.31 - 13.896.623.000.136\n01. IDENTIFICA\u00c7\u00c3O\nNOME EMPRESARIAL (firma ou denomina\u00e7\u00e3o)\nN\u00b0 DE INSCRI\u00c7\u00c3O NO CNPJ\nRF MOTORS COMERCIO DE VEICULOS LTDA.\n13.896.623/0001-36\n02. MOTIVO DO PREENCHIMENTO\nRELA\u00c7\u00c3O DOS EVENTOS SOLICITADOS / DATA DO EVENTO\n203 Exclus\u00e3o do t\u00edtulo do estabelecimento (nome de fantasia) - 12/01/2015\n211 Altera\u00e7\u00e3o de endere\u00e7o dentro do mesmo munic\u00edpio - 12/01/2015\n220 Altera\u00e7\u00e3o do nome empresarial (firma ou denomina\u00e7\u00e3o) - 12/01/2\n03. IDENTIFICA\u00c7\u00c3O DO REPRESENTANTE DA PESSOA JUR\u00cdDICA\nNOME\nBRUNO VINICIUS FERREIRA\nCPF\n340.446.998-44\nILOCAL\nDATA\n12/01/2015\n04. C\u00d3DIGO DE CONTROLE DO CERTIFICADO DIGITAL\nEste documento foi assinado com uso de senha da Sefaz SP\nAprovado pela Instru\u00e7\u00e3o Normativa RFB n\u00ba 1.183, de 19 de agosto de 2011\n12/01/2015\nhttp://www.receita fazenda.gov.br/pessoajuridica/cnpj/fcpj/dbe.asp\n\n\nES\nSP\nGOVERNO DO ESTADO DE S\u00c3O BAULO\nSECRETARIA DE DESENVOLVIMENTO ECONOMICO, CIENCIA E TECNOLOGIA\nJUNTA COMERCIAL DO ESTADO.DE S\u00c3O PAULO: JUCES...\nJUCESP\nAnta Comercial do\nEstado de Sio Pub\nDECLARA\u00c7\u00c3O\n,\nEu, Bruno Vinicius Ferreira, portador da C\u00e9dula de Identidade n\u00ba 42318703-X, inscrito no\nCadastro de Pessoas F\u00edsicas - CPF sob n\u00ba 340.446.998-44, na qualidade de titular, s\u00f3cio ou\nrespons\u00e1vel legal da empresa RF MOTOR'S Com\u00e9rcio de ve\u00edculos Ltda. - ME, DECLARO\nestar ciente que o ESTABELECIMENTO situado no(a) Avenida Regente Feij\u00f3, 277 Vila\nRegente Feij\u00f3, S\u00e3o Paulo, S\u00e3o Paulo, CEP 03342-000, N\u00c3O PODER\u00c1 EXERCER suas\natividades sem que obtenha o parecer municipal sobre a viabilidade de sua instala\u00e7\u00e3o e\nfuncionamento no local indicado, conforme diretrizes estabelecidas na legisla\u00e7\u00e3o de uso e\nocupa\u00e7\u00e3o do solo, posturas municipais e restri\u00e7\u00f5es das \u00e1reas de prote\u00e7\u00e3o ambiental, nos\ntermos do art. 24, $2 do Decreto Estadual n\u00ba 55.660/2010 e sem que tenha um CERTIFICADO\nDE LICENCIAMENTO INTEGRADO V\u00c1LIDO, obtido pelo sistema Via R\u00e1pida Empresa\nM\u00f3dulo de Licenciamento Estadual.\nDeclaro ainda estar ciente que qualquer altera\u00e7\u00e3o no endere\u00e7o do estabelecimento, em sua\natividade ou grupo de atividades, ou em qualquer outra das condi\u00e7\u00f5es determinantes \u00e0\nexpedi\u00e7\u00e3o do Certificado de Licenciamento Integrado, implica na perda de sua validade,\nassumindo, desde o momento da altera\u00e7\u00e3o, a obriga\u00e7\u00e3o de renov\u00e1-lo.\nPor fim, declaro estar ciente que a emiss\u00e3o do Certificado de Licenciamento Integrado poder\u00e1\nser solicitada por representante legal devidamente habilitado, presencialmente e no ato da\nretirada das certid\u00f5es relativas ao registro empresarial na Prefeitura, ou pelo titular, s\u00f3cio, ou\ncontabilista vinculado no Cadastro Nacional da Pessoa Jur\u00eddica (CNPJ) diretamente no site da\nJucesp, atrav\u00e9s do m\u00f3dulo de licenciamento, mediante uso da respectiva certifica\u00e7\u00e3o digital.\nBruno Vinicius Ferreira\nRG: 42318703-X\nRF MOTOR'S Com\u00e9rcio de ve\u00edculos Ltda. - ME"   
        
    context_content = 'position_token'
    context_content = 'windows_token'
    use_sentence_id = True
    window_overlap = 0.5
    max_windows = 3
    
    start_position = 158
    max_size = 200

    #tokenizer = AutoTokenizer.from_pretrained('models/', do_lower_case=False)
    tokenizer = AutoTokenizer.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab', do_lower_case=False)
    max_tokens = 150
    question =  'Qual o tipo, a classe, o órgão emissor, a localização e a abrangência?'

    context, offset = get_context(
        document,
        context_content=context_content,
        max_size=max_size,
        start_position=start_position,
        proportion_before=0.2,
        return_position_offset=True,
        use_sentence_id=use_sentence_id,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        max_windows=max_windows,
        question=question,
        window_overlap=window_overlap,
        verbose=True)

    print('--> testing the offset:')
    if isinstance(context, list):
        context, offset = context[-1], offset[-1]  # last window
    print('>>>>>>>>>> using the offset\n' + document['text'][offset:offset + len(context)])
    print('>>>>>>>>>> returned context\n' + context)


if __name__ == "__main__":
    main()
