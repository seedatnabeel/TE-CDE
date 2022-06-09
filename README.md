# Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations


This repository contains the implementation of TE-CDE. A method to model counterfactual outcomes using neural controlled differential equations (CDEs).

For more details, please read our [ICML 2022 paper](LINK): 'Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations'.

## Installation
1. Clone the repository
2. Create a new virtual environment with Python 3.7. e.g:
```shell
    virtualenv tecde_env
```
3. Run the following command from the repository directory:
 ```shell
pip install -r requirements.txt
 ```
This should install all required packages.

## Usage
The code is run through a command line interface which can be parameterized as described below.

All the relevant code can be found in the src folder.

Note: All intermediate outputs and results are logged to [Weights and Biases - wandb](https://wandb.ai) by default. An account is required and the WANDB_API_KEY need to be set in the environment variables or via os.environ in main.py. A wandb entity also needs to be set.

## Parameters:

```
python main.py
[--chemo_coeff [chemotherapy coefficient]]
[--radio_coeff [radiotherapy coefficient]]
[--results_dir [path to directory to store results]]
[--model_name [final model name]]
[--load_dataset [boolean whether to load a saved version of the dataset from file]]
[--experiment [name of experiment yml]]
[--data_path [path to experiment data if loading one from a location]] [--use_transformed [boolean whether to use an already trasnformed version of the dataset to speed things up]]
[--multistep [boolean whether to run the multistep function]]
[--kappa [kappa parameter for the Hawkes process]]
[--lambda_val [max range of the lambda value for the adversarial training]][--max_horizon [maximum step horizon]]
[--save_raw_datapath [path to save the raw dataset, so it can be reused and speed things up]]
[--save_transformed_datapath [path to save the transformed dataset, so it can be reuse and speed things up]]
```

Note the most important parameters is   ``--chemo_coeff``, ``--radio_coeff``and ``--kappa``


## Example usage:

Below is an example where we load the dataset from a mounted Google drive folder. Of course results are variable depending on the dataset generated and the transformation.

```
python main.py --data_path "/content/drive/MyDrive/Data/new_cancer_sim_4_4.p" --chemo_coeff=4 --radio_coeff=4 --kappa=1 --multistep=True
```

## Data generation:
We suggest generating a dataset and transforming it for ease of re-use. This can be done using these two commands ``[--save_raw_datapath]``and ``[--save_transformed_datapath]``, as well as, setting ``--use_transformed=False`` and not supplying a ``--data_path``. After which these datasets can be re-used to speed things up. Note that we have left the dataset generation as random, such that when you re-run it one can always get a different dataset. This might result in some variability in results depending on the dataset. Note these files by default are pickles and we provide write and read functions in ``src/utils/data_utils.py``.

We suggest then mounting these datasets (raw or transformed) through Google Drive - where you can read the pickle by passing the ``--data_path``.
