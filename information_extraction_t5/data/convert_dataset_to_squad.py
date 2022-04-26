"""Converts the dataset into SQuAD format."""
import json
import os
from typing import List, Tuple

import configargparse
import numpy.random as nr
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import information_extraction_t5.data.basic_to_squad as basic_to_squad
from information_extraction_t5.data.file_handling import load_raw_data
from information_extraction_t5.features.preprocess import get_all_qa_ids

DATA_VERSION = "0.1"


def convert_raw_data(documents: List[tuple],
                     project: str,
                     all_qa_ids: List[str],
                     tokenizer: AutoTokenizer,
                     choose_question: str,
                     use_sentence_id: bool,
                     args) -> Tuple[List[dict], int]:
    """Loops over the documents and converts to SQuaD format.

    Args:
        documents: list with selected document tuples
        project: the project name
        tokenizer: T5 Tokenizer instance
        choose_question: flag to indicate which questions to use
        is_true: True for training data (useful for function build_answer)
        args: additional configs
    """
    qa_data = []
    qa_counter = 0

    for doc_id, document in documents:
        document['uuid'] = doc_id
        document_data, count = convert_document(
            document,
            project=project,
            all_qa_ids=all_qa_ids,
            max_size=args.max_size,
            type_names=args.type_names,
            use_compound_question=args.use_compound_question,
            return_raw_text=args.return_raw_text,
            context_content=args.context_content,
            window_overlap=args.window_overlap,
            max_windows=args.max_windows,
            tokenizer=tokenizer,
            max_tokens=args.max_seq_length,
            choose_question=choose_question,
            use_sentence_id=use_sentence_id)
        qa_counter += count

        # To finish a document, include its document_data into the
        # qa_json
        if count > 0:
            qa_data.append(document_data)

    return qa_data, qa_counter


def convert_document(document,
                     project='publicacoes',
                     all_qa_ids=['publicacoes.tipoPublicacao'],
                     max_size=4000,
                     type_names=None,
                     use_compound_question=None,
                     return_raw_text=None,
                     context_content='abertura',
                     window_overlap=0.5,
                     max_windows=3,
                     tokenizer=None,
                     max_tokens=512,
                     choose_question='first',
                     use_sentence_id: bool = False):
    """Converts a document and returns it along with the question count."""
    if return_raw_text is None:
        return_raw_text = []
    if use_compound_question is None:
        use_compound_question = []
    if type_names is None:
        type_names = []

    document_data, count = basic_to_squad.get_document_data(
        document,
        document_type=project,
        all_qa_ids=all_qa_ids,
        max_size=max_size,
        list_of_use_compound_question=use_compound_question,
        list_of_type_names=type_names,
        context_content=context_content,
        window_overlap=window_overlap,
        max_windows=max_windows,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        choose_question=choose_question,
        use_sentence_id=use_sentence_id)

    return document_data, count


def main():
    """Preparing data for QA in SQuAD format."""
    parser = configargparse.ArgParser(
        'Preparing data for QA',
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', '--my-config', required=True,
                        is_config_file=True,
                        help='config file path')

    parser.add_argument('--project', action='append', required=True,
                        help='List pointing out the project each train/test '
                             'dataset came from')
    parser.add_argument('--raw_data_file', action='append', required=True,
                        help='List of raw train datasets to use in the '
                             'experiment')
    parser.add_argument('--raw_valid_data_file', action='append',
                        help='List of raw validation datasets to use in the '
                             'experiment')
    parser.add_argument('--raw_test_data_file', action='append',
                        help='List of raw test datasets to use in the '
                             'experiment')
    parser.add_argument('--train_file', type=str,
                        default='data/interim/train-v0.1.json')
    parser.add_argument('--valid_file', type=str,
                        default='data/interim/dev-v0.1.json')
    parser.add_argument('--test_file', type=str,
                        default='data/interim/test-v0.1.json')
    parser.add_argument('--type_names', nargs='+', default=['matriculas.imovel'],
                        help='List of first-level chunks (qa_id) to use in the '
                        'experiment')
    parser.add_argument('--use_compound_question', nargs='+',
                        default=['matriculas.area_terreno_comp'],
                        help='List of fields (qa_id) that must use '
                        'compound question gathering all nested information '
                        'in answer (instead of per-subchunk questions)')
    parser.add_argument('--return_raw_text', nargs='+', default=['estado'],
                        help='List of fields (type_name) that '
                        'require both canonical answer and how it appears in '
                        'the text. Valid to individual and compound questions. NOT IMPLEMENTED.')

    parser.add_argument("--valid_percent", default=0.2, type=float,
                        help='Percentage of dataset to used as validation')
    parser.add_argument("--max_size", default=1024, type=int,
                        help="The maximum input length after char-based "
                             "tokenization. And also the maximum context size "
                             "for char-based contexts.")
    parser.add_argument("--context_content", type=str, default='abertura',
                        help="Definition of context content for generic "
                             "type-names (max_size, position, token, "
                             "position_token, windows, or windows_token)")
    parser.add_argument("--train_choose_question", type=str, default='all',
                        help='Choose which question of the list to use for '
                             'training set (first, random, all). '
                             'Validation/test set use first.')
    parser.add_argument('--train_force_qa', action="store_true",
                        help='Set this flag if you want to force not-applicable '
                            'qas for qa_ids that does not exist in the document. '
                            'This is required for test set.')
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for choose qestion")

    # used to get contexts
    parser.add_argument("--model_name_or_path", default='t5-small', type=str,
                        help="Path to pretrained model or model identifier "
                        "from huggingface.co/models")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same "
                        "as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the "
                        "same as model_name")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased "
                        "model.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after "
                        "WordPiece tokenization. Sequences longer than this "
                        "will be truncated, and sequences shorter than this "
                        "will be padded.")
    parser.add_argument("--window_overlap", default=0.5, type=float,
                        help="Define the overlapping of sliding windows.")
    parser.add_argument("--max_windows", default=3, type=int,
                        help="the maximum number of windows to generate, use -1 "
                        "to get all the possible windows.")
    parser.add_argument("--use_sentence_id", action="store_true",
                        help="Set this flag if you are using the approach that "
                        "breaks the contexts into sentences.")

    args, _ = parser.parse_known_args()

    assert len(args.project) == len(args.raw_data_file) == \
        len(args.raw_valid_data_file) == len(args.raw_test_data_file), \
        ('raw_data_file, raw_valid_data_file and raw_test_data_file lists '
         'must have same size of projects list')
    assert args.train_choose_question in ['first', 'random', 'all'], \
        ('train_choose_question must be "first", "random" or "all"')
    assert args.context_content in ['max_size', 'position', 'token',
                                    'position_token', 'windows', 'windows_token'], \
        ('context_content must be "max_size", "position", "token", "position_token", '
        '"windows" or "windows_token"')

    # set tokenizer for context_context based on tokens
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name
        else args.model_name_or_path,
        use_fast=False,
        do_lower_case=args.do_lower_case
    )

    # setting seed for choose question
    nr.seed(args.seed)

    print('>> Using the following fields with respective compound-qa indicator:')
    for type_name in args.type_names:
        print(f'- {type_name:<43} {type_name in args.use_compound_question}\t')
    print(f'>> List of fields that require how answer appears in '
          f'the text: {args.return_raw_text}')

    qa_train_json = {'data': [], 'version': DATA_VERSION}
    qa_valid_json = {'data': [], 'version': DATA_VERSION}
    qa_test_json = {'data': [], 'version': DATA_VERSION}

    train_qa_counter, valid_qa_counter, test_qa_counter = 0, 0, 0

    for (raw_data_file, raw_valid_data_file, raw_test_data_file, project) in \
            zip(args.raw_data_file, args.raw_valid_data_file,
                args.raw_test_data_file, args.project):

        print('\n')

        # Extract the list of all possible qa_ids for the current document class.
        # This forces N/A qas for valid/test, and for train if --train_force_qa
        all_qa_ids = get_all_qa_ids(
            document_class=project,
            list_of_type_names=args.type_names,
            list_of_use_compound_question=args.use_compound_question)

        # prepare VALIDATION set (if provided)
        has_valid_set = raw_valid_data_file is not None \
            and raw_valid_data_file != 'None'

        if has_valid_set:

            print(f'>> Loading the VALID dataset {raw_valid_data_file} '
                  f'({project})...')
            _, all_documents, raw_data_fname = load_raw_data(
                raw_valid_data_file
            )

            print(f'>> Converting the VALID dataset {raw_valid_data_file} '
                  'into SQuAD format...')
            qa_data, qa_counter = convert_raw_data(
                documents=all_documents,
                project=project,
                all_qa_ids=all_qa_ids,
                tokenizer=tokenizer,
                choose_question='first',
                use_sentence_id=args.use_sentence_id,
                args=args
            )

            if qa_counter > 0:
                print(f'{raw_valid_data_file} (valid) dataset has '
                      f'{qa_counter} question-answers')
                valid_qa_counter += qa_counter
                qa_valid_json['data'].extend(qa_data)

            if raw_valid_data_file.endswith('tar') \
                    or raw_valid_data_file.endswith('tar.gz'):
                os.unlink(raw_data_fname)

        has_test_set = raw_test_data_file is not None \
            and raw_test_data_file != 'None'

        # prepare TEST set (if provided)
        if has_test_set:

            print(f'>> Loading the TEST dataset {raw_test_data_file} '
                  f'({project})...')
            _, all_documents, raw_data_fname = load_raw_data(
                raw_test_data_file
            )

            print(f'>> Converting the TEST dataset {raw_test_data_file} into '
                  'SQuAD format...')
            qa_data, qa_counter = convert_raw_data(
                documents=all_documents,
                project=project,
                all_qa_ids=all_qa_ids,
                tokenizer=tokenizer,
                choose_question='first',
                use_sentence_id=args.use_sentence_id,
                args=args
            )

            if qa_counter > 0:
                print(f'{raw_test_data_file} (test) dataset has '
                      f'{qa_counter} question-answers')
                test_qa_counter += qa_counter
                qa_test_json['data'].extend(qa_data)

            if raw_test_data_file.endswith('tar') \
                    or raw_test_data_file.endswith('tar.gz'):
                os.unlink(raw_data_fname)

        # prepare TRAIN set
        print(f'>> Loading the dataset {raw_data_file} ({project})...')
        _, all_documents, raw_data_fname = load_raw_data(
            raw_data_file
        )

        if not has_valid_set and 0 < args.valid_percent < 1.0:
            documents_train, documents_valid = train_test_split(
                all_documents,
                test_size=args.valid_percent,
                random_state=42)

            qa_data, qa_counter = convert_raw_data(
                documents=documents_valid,
                project=project,
                all_qa_ids=all_qa_ids,
                tokenizer=tokenizer,
                choose_question='first',
                use_sentence_id=args.use_sentence_id,
                args=args
            )

            # if a TEST dataset is provided, use the split for VALIDATION only,
            # otherwise, use it for both VALIDATION and TEST
            if has_test_set:
                if qa_counter > 0:
                    print(f'{raw_data_file} (valid) dataset has {qa_counter} '
                          f'question-answers')
                    valid_qa_counter += qa_counter
                    qa_valid_json['data'].extend(qa_data)
            else:
                if qa_counter > 0:
                    print(f'{raw_data_file} (valid/test) dataset has '
                          f'{qa_counter} question-answers')
                    valid_qa_counter += qa_counter
                    qa_valid_json['data'].extend(qa_data)
                    test_qa_counter += qa_counter
                    qa_test_json['data'].extend(qa_data)

        else:
            documents_train = all_documents

        print(
            f'>> Converting the dataset {raw_data_file} into SQuAD format...'
        )
        qa_data, qa_counter = convert_raw_data(
            documents=documents_train,
            project=project,
            all_qa_ids=all_qa_ids if args.train_force_qa else [],
            tokenizer=tokenizer,
            choose_question=args.train_choose_question,
            use_sentence_id=args.use_sentence_id,
            args=args
        )
        print(f'{raw_data_file} (train) dataset has {qa_counter} '
              f'question-answers')
        train_qa_counter += qa_counter
        qa_train_json['data'].extend(qa_data)

        if raw_data_file.endswith('tar') or raw_data_file.endswith('tar.gz'):
            os.unlink(raw_data_fname)

    print(f'\nTRAIN dataset has {train_qa_counter} question-answers')
    print(f'VALID dataset has {valid_qa_counter} question-answers')
    print(f'TEST dataset has {test_qa_counter} question-answers')

    # Save the train, valid and test processed data
    os.makedirs(os.path.dirname(args.train_file), exist_ok=True)
    with open(args.train_file, 'w', encoding='utf-8') as outfile:
        json.dump(qa_train_json, outfile)
    with open(args.valid_file, 'w', encoding='utf-8') as outfile:
        json.dump(qa_valid_json, outfile)
    with open(args.test_file, 'w', encoding='utf-8') as outfile:
        json.dump(qa_test_json, outfile)


if __name__ == "__main__":
    main()
