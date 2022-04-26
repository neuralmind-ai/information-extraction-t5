# Information Extraction using T5

This project provides a solution for training and validating seq2seq models for information extraction. The method can be applied for any text-only type of document, such as legal, registration, news, etc. The approach used here for IE is question answering.

In contrast to typical approachs, this project ...


# Installation

Clone the repository and install by running:
        
        pip install .

# Fine-tuning

Configure the parameters in *params.yaml*. Preprocess the datasets by running:

        python information_extraction_t5/data/convert_dataset_to_squad.py -c params.yaml

Run the experiment:

        python information_extraction_t5/main.py -c params.yaml

Note that, for now, only the dataset NM-Publications is available. To extend for new datasets, please consult the section xxx.

# Inference

Just run:

        python information_extraction_t5/predict.py -c params.yaml

This will result in N output files. TODO: describe them...

# Setting the hyperparameters

Plase, run ```python information_extraction_t5/main.py --help``` to know all the the settings related to data pre-processing, training, inference and result post-processing.

TODO: describe the arguments

Main arguments to describe:
- context_content
- ...

# Extending for new datasets

blablabla

TODO: Give a name to the dataset

TODO: Describe how the format the dataset must be converted to.

TODO: Present the basic_to_squad we develop for datasets without text markings (start, end position) and without canonical format.

This code that converts basic dataset to squad format supports compound QAs, but does not support sentence-ids and canonical format. In or
der to use those features for your own dataset, you must convert the data into squad format by your own.

TODO: Describe the expected JSON format for basic_to_squad with an example. Show compound chunk.

TODO: List train, validation, test. Mention the valid_percent (maybe this should be included in the section before).

TODO: Describe how to set the questions and the type-map. Show the format of the question dictionary.

* The model questions are available [here](information_extraction_t5/features/questions).


# Cite as
       
```bibtex 
@inproceedings{pires2022sequence,
          title = {Sequence-to-Sequence Models for Extracting Information from Registration and Legal Documents},
          author = {Pires, Ramon and de Souza, Fábio C. and Rosa, Guilherme and Lotufo, Roberto A. and Nogueira, Rodrigo},
          publisher = {arXiv},
          doi = {10.48550/ARXIV.2201.05658},
          url = {https://arxiv.org/abs/2201.05658},
          year = {2022},
        }
```

        
