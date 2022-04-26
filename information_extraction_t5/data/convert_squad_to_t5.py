"""Converts the dataset from SQuAD format to T5 format."""
import torch
from rich.progress import track
from typing import List, Union

from transformers.data.processors.squad import SquadExample

from information_extraction_t5.features.preprocess import generate_t5_input_sentence, generate_t5_label_sentence
from information_extraction_t5.utils.balance_data import balance_data

class QADataset(torch.utils.data.Dataset):
    """
    Dataset for question-answering.

    Args:
        examples: the inputs to the model in T5 format.
        labels: the targets in T5 format.
        document_ids: the IDs to reference specific documents.
        example_ids: the IDs to reference specific pairs dataset-field.
        negative_ratios: the resultant negative-positive ratio of the samples.
        return_ids: indicates if the dataset will return the document-ids and example_ids.
    
    Returns:
        Dataset

    Ex.:
    examples    = ['question: When was the Third Assessment Report published? context: Another example of scientific research ...']
    labels      = ['2011']
    document_ids= ['ec57d59d-972c-40fc-82ff-c7c818d7dd39']
    example_ids = ['reports.third_assessment.publication_data']
    """

    def __init__(self, examples, labels, document_ids, example_ids, negative_ratio=1.0, return_ids=False):
        if negative_ratio >= 1.0:
            self.examples, self.labels, self.document_ids, self.example_ids = balance_data(
                examples, labels, document_ids, example_ids, negative_ratio=negative_ratio
            )
        else:
            self.examples = examples
            self.labels = labels
            self.document_ids = document_ids
            self.example_ids = example_ids
        self.return_ids = return_ids

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        if self.return_ids:
            return self.examples[idx], self.labels[idx], self.document_ids[idx], self.example_ids[idx]
        else:
            return self.examples[idx], self.labels[idx]
            

def squad_convert_examples_to_t5_format(
    examples: List[SquadExample],
    use_sentence_id: bool = True,
    evaluate: bool = False,
    negative_ratio: int = 0,
    return_dataset: Union[bool, str] = False,
    tqdm_enabled: bool = True,
):
    """Converts a list of examples into a list to the T5 format for
        question-answer with prefix question/context.

        Args:
            examples: examples to convert to T5 format.
            evaluate: True for validation or test dataset.
            negative_ratio: balances dataset using negative-positive ratio.
            return_dataset: if True, returns a torch.data.TensorDataset.
            tqdm_enabled: if True, uses tqdm.

        Returns:
            list of examples into a list to the T5 format for
        question-answer with prefix question/context.

        Examples:
            >>> processor = SquadV2Processor()
            >>> examples = processor.get_dev_examples(data_dir)
            >>> examples, labels = squad_convert_examples_to_t5_format(
            >>>     examples=examples)
    """

    examples_t5_format = []
    labels_t5_format = []
    document_ids = []   # which document the example came from? (e.g, 54f94949-0fb4-45e5-81dd-c4385f681e2b)
    example_ids = []    # which document-type and type-name does the example belong to? (e.g., matriculas.endereco)

    for example in track(examples, description="convert examples to T5 format", disable=not tqdm_enabled):

        # prepare the input
        x = generate_t5_input_sentence(example.context_text, example.question_text, use_sentence_id)
        
        # extract answer and start position (squad-example is in evaluate mode)
        y = example.answers[0]['text']  # getting the first answer in the list
        answer_start = example.answers[0]['answer_start']

        # prepate the target
        y = generate_t5_label_sentence(y, answer_start, example.context_text, use_sentence_id)
        
        examples_t5_format.append(x)
        labels_t5_format.append(y)
        document_ids.append(example.title)
        example_ids.append(example.qas_id)

    if return_dataset:
        # Create the dataset
        dataset = QADataset(examples_t5_format, labels_t5_format, document_ids, 
            example_ids, negative_ratio=negative_ratio, return_ids=evaluate)

        return examples_t5_format, labels_t5_format, dataset
    else:
        return examples_t5_format, labels_t5_format