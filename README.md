# Information Extraction using T5

 [![arXiv](https://img.shields.io/badge/arXiv-2101.05658-f9f107.svg)](https://arxiv.org/abs/2201.05658)

This project provides a solution for training and validating seq2seq models for information extraction. The method can be applied for any text-only type of document, such as legal, registration, news, etc. The project extracts information by question answering.

In this work, we evaluate sequence-to-sequence models as an alternative to token-level classification methods to extract information from documents. T5 models are finetuned to jointly extract the information and generate the output in a structured format. Post-processing steps are learned during training, eliminating the need for rule-based methods and simplifying the pipeline.

Neither the models weights nor the datasets are available for ethical issues. But we make efforts to release the source code that works for different models of T5 family, and can be easily extended for new datasets and different languages.

# Installation

Clone the repository and install by running:

```bash
pip install .
```

# Fine-tuning

Configure the parameters in `params.yaml`. Preprocess the datasets by running:

```bash
python information_extraction_t5/data/convert_dataset_to_squad.py -c params.yaml
```

Start the fine-tuning experiment:

```bash
python information_extraction_t5/train.py -c params.yaml
```

Note that, for now, only one [tiny synthetic dataset](data/raw/sample_train.json) is available. To extend for new datasets, please consult [this section](#extending-for-new-datasets).

# Inference

For running inference, just execute:

```bash
python information_extraction_t5/predict.py -c params.yaml
```

After finising the prediction, some metrics and post-processed outputs files will be generated:

- `metrics_by_typenames.json`: JSON file with exact matching and F1-score for each *field*, each dataset and all the documents.
- `metrics_by_documents.json`: JSON file with exact matching and F1-score for each *document*, each dataset and all the documents.
- `outputs_by_typenames.txt`: TXT file with labels, predictions, *document id*, probability of selected window, and window id, grouped by *fields*.
- `outputs_by_documents.txt`: TXT file with labels, predictions, *field id*, probability of selected window, and window id, grouped by *documents*.
- `output_sheet.xlsl`: Excel file with field id, labels, predictions and probs grouped by documents.
- `output_sheet_client.xlsl`: Excel file with labels, predictions, probs and metrics grouped by dataset.

# Setting the hyperparameters

To know all the settings related to pre-processing, training, inference and pos-processing stages, please run:

```bash
python information_extraction_t5/train.py --help
```

You will find an extensive list of parameters because it is inheriting Pytorch-Lightning's Trainer arguments.
Give special attention only to the parameters that are in the `params.yaml` file.

<!-- TODO: describe the most import arguments -->
<!-- Main arguments to describe: -->
<!-- - context_content -->

# Extending for new datasets

In this section we explain how to include new datasets for running fine-tuning and inference.
It is important to emphasize that the four datasets that have been originally applied in the project cannot be released for ethical issues.

## Preparing the questions and type-map

There are two preliminar steps when extending the project for new datasets.

### Mapping field names to clues *[Mandatory]*

The original field names of the datasets can be noisy, not natural. One important step is converting those irregular names into natural ones. The natural names will be used as clues in the answers.

For each dataset, it is necessary to [map](information_extraction_t5/features/questions/type_map.py) field names (we call it as type-name in the code) to types and vice-versa. The types are used as clues in brackets in T5 outputs. The field names are recovered in post-processing stage.

Each dataset has it own `TYPENAME_TO_TYPE` dictionary. We strongly recommend that the types used in all the projects be consistent, and as generic as possible. For example, using *CPF/CNPJ* for all CPFs and CNPJs, regardless of being a consultant, current account holder, business partner, land owner, etc.

### Formulating questions *[Optional]*

If your dataset does not follow the required [SQuAD format](#format-of-the-dataset) and you intend to use the [pre-processing code](#converting-the-dataset-to-squad-format), before starting the conversion it is necessary to formulate the [questions](information_extraction_t5/features/questions/questions.py).

Each dataset will have a particular dictionary of questions, in which the key is the field name (we call it as type-name in the code) and the value is a list of questions. 

HINT: We use one list of questions for each field as a strategy to augment the dataset. You can use the data augmentation by setting `train_choose_question: all` in `params.yaml`. Use `random` to select one question randomly, or `first` to get the first one for each field.

If you have a compound information (the value is an internal dictionary), we recommend representing the dict as an OrderedDict in order to use the dictionary keys as field signature, ensure a possíble compound answer will have it sub-answers in an inmutable order.

## Format of the dataset

As the project aims at extracting information using QA modality, we adopt the SQuAD as the format of the datasets, with a few adaptations. Below we present an example to illustrate the structure of the dataset file and describe the adaptations to enable the use of sliding windows and the reference of each pair [document, field], in order to enable an effective metric computation for each document, dataset and field.

```json
{
  "data": [
    {
      "title": "318",
      "paragraphs": [
        {
          "context": "Proposta de Abertura de Conta, Contrata\u00e7\u00e3o de Cr\u00e9dito e\nAdes\u00e3o a Produtos e Servi\u00e7os Banc\u00e1rios - Pessoa F\u00edsica\nID00147\nAg\u00eancia N\u00ba\n1234\nConta Corrente 0011-2347-0000809875312\nCondi\u00e7\u00e3o de Movimenta\u00e7\u00e3o da Conta X Individual\nAltera\u00e7\u00e3o cadastral\nAngariador (matr\u00edcula) L\n00098961\nDados B\u00e1sicos do Titular\nCPF\n516.759.760-90\n...",
          "qas": [
            {
              "answers": [
                {
                  "answer_start": 157,
                  "text": "[Ag\u00eancia]: 2347"
                }
              ],
              "question": "Qual \u00e9 o n\u00famero da ag\u00eancia?",
              "id": "form.agencia"
            },
            {
              "answers": [
                {
                  "answer_start": -1,
                  "text": "[Nome]: N/A"
                }
              ],
              "question": "Qual \u00e9 o nome?",
              "id": "form.nome_completo"
            }
          ]
        },
        {
          "context": "...\nNome Completo ANA MADALENA SILVEIRA ALVES\nDocumento de Identifica\u00e7\u00e3o CNH CTPS Entidade de Classe Mercosul Passaporte\nProtocolo Refugiado\nRIC RNE\nCIE Guia de Acolhimento ao Menor Registro Nacional Migrat\u00f3rio\nN\u00b0 Documento / N\u00b0 da S\u00e9rie (CTPS)\n73258674 \u00d3rg\u00e3o Emissor SSP\nUF BA\nData de Emiss\u00e3o 21/07/2018 Data de Vcto (passaporte/CNH).",
          "qas": [
            {
              "answers": [
                {
                  "answer_start": -1,
                  "text": "[Ag\u00eancia]: N/A"
                }
              ],
              "question": "Qual \u00e9 o n\u00famero da ag\u00eancia?",
              "id": "form.agencia"
            },
            {
              "answers": [
                {
                  "answer_start": 18,
                  "text": "[Nome]: ANA MADALENA SILVEIRA ALVES"
                }
              ],
              "question": "Qual \u00e9 o nome?",
              "id": "form.nome_completo"
            }
          ]
        }
      ]
    }
  ],
  "version": "0.1"
}
```
The example we presented herein includes one document whose `id = 318`, and context fits into two sliding windows. We adapted SQuAD format by transforming the list of different documents related to the same theme into a list of different sliding windows of the same document. For each document, reference in `title`, `paragraphs` is a list of dictionaries that have context and an internal dictionary of QAs.

The dictionaries of QAs follows the same intuition of SQuAD dataset, but we included in `id` the signature of QA, that involves the project (dataset name) and the field. This is very important since enables to get metrics not only for the datasets altogether, but also for each dataset individually as well as for each field. 

## Adding the dataset

Assuming you have a dataset already pre-processed, in SQuAD format, to include it in the project you may choose a name for it and edit the following parameters in the `params.yaml` file:

```yaml
project: [
  form,
  ]
train_file: data/processed/train-v0.1.json
valid_file: data/processed/dev-v0.1.json
test_file: data/processed/test-v0.1.json
```

Note that it's possible to include several datasets in the list of projects, but each `{train, valid, test}_file` includes the examples of all the datasets listed in `project`.

## Converting the dataset to SQuAD format

If your dataset is not in the complex SQuAD-like format with the document divided into sliding windows, the pairs of question-answers, the correct qa-id, don't worry! We are releasing a code to [convert the dataset to the expected format](information_extraction_t5/data/basic_to_squad.py).

What you need to do is just ensure the dataset follows the format of a basic JSON: a dictionary of documents, in which each key is the document-id, and each value is an internal dict that must have the key "text" with the respective document content as value, and other pairs key-values representing the fields the document has.

You can visualize [here](data/raw/sample_train.json) one raw dataset that is ready to be converted into SQuAD format. NOTE: If you want to extract compound information (using compound QA feature) for the one compound field, such as `address`, the value of the respective key must be another dictionary with the expected information.

Thus, to generate a SQuAD-like dataset illustrated in the previous subsection, just set the parameters as below:

```yaml
project: [
  form,
  ]
raw_data_file: [
  data/raw/sample_train.json,
  ]
raw_valid_data_file: [
  null,
  ]
raw_test_data_file: [
  data/raw/sample_test.json,
  ]
train_file: data/processed/train-v0.1.json
valid_file: data/processed/dev-v0.1.json
test_file: data/processed/test-v0.1.json
type_names: [
  form.agencia,
  form.nome_completo,
  ]
```

The parameter names are intuitive. You can include any number of dataset names and their respected train, validation and test paths (the four lists must have the same number of parameters). If any of the datasets does not have a validation subset, just include `null` in the position, and a fraction of `valid_percent` of the training set will be moved for validation set.

Finally, just run the command below to get the listed datasets converted to SQuAD format and saved as `{train, valid, test}_file`.

```bash
python information_extraction_t5/data/convert_dataset_to_squad.py -c params.yaml
```

### Limitation

The released code for dataset pre-processing does not include the features `sentence-ids` and `raw-text` formats as it would require more complex and ellaborated raw dataset, whose structure must include annotations of positions and texts both in raw and canonical formats. Those features are important only for industrial applications, but, depending of the dataset size, you can manually include `answer_start` for each qa, and setting the answer as *N/A* if it does not fit in the window. For training the model to extract canonical and raw-text information, you can change both the questions and answers as:

```
Q: What is the state?
A: [State]: São Paulo

Q: What is the state and how does it appear in the text?
A: [State]: SP [appears in the text]: São Paulo
```

# Cite as
       
```bibtex 
@inproceedings{pires2022seq2seq,
          title = {Sequence-to-Sequence Models for Extracting Information from Registration and Legal Documents},
          author = {Pires, Ramon and de Souza, Fábio C. and Rosa, Guilherme and Lotufo, Roberto A. and Nogueira, Rodrigo},
          publisher = {arXiv},
          doi = {10.48550/ARXIV.2201.05658},
          url = {https://arxiv.org/abs/2201.05658},
          year = {2022},
        }
``` 
