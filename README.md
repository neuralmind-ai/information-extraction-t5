# Question Answering T5 for documents

This library provides methods and requirements to train and test a 
[T5 model](https://arxiv.org/pdf/1910.10683.pdf) for question answering of 
*Matrículas*, *Certidões*, *Publicações* and *PAC*.

* The model questions are available [here](src/features/questions).

* Follows the [DataScience Cookiecutter directory structure](https://drivendata.github.io/cookiecutter-data-science/).


# Installation

Clone the repository and install its dependencies according to the usecase.
        
        # Training
        pip install .[dev]
        
        # Deployment
        pip install .[deploy]
        
        # Minimal (only used by the deploy container)
        pip install .

# How to Train

Configure the parameters in *params.yaml*. Preprocess the datasets by running:

        python src/data/convert_dataset_to_squad.py -c params.yaml

Run the experiment:

        python src/main.py -c params.yaml

For running with A100 gpu, install torch using:

	pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.8.1+cu111
