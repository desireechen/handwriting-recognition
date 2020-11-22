# Handwriting Recognition

This repository is adapted from the labs in [Full Stack Deep Learning](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project). I am adding more of my learning notes and experiments over time.

## Datasets

[EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)

[IAM Lines](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

The data is downloaded to `data/raw` folder. The processed data is stored in `data/processed` folder. Due to the large dataset sizes, the `metadata.toml` file contains the remote source of the data. From this file, one is able to obtain the dataset. This works somewhat similar to [Data Version Control](https://dvc.org/) and [Git Large File Storage](https://git-lfs.github.com/). 

## Models

`CharacterModel` class works on image datasets with one-hot labels.

`LineModel` class recognizes text in an image of a handwritten line of text. 

`LineModelCtc` class recognizes text in an image of a handwritten line of text. The model uses Connectionist Temporal Classification (CTC). In the case of handwriting recognition, I am using Long short-term memory (LSTM) networks. 

## Networks

4 main networks: Multi-layer perceptron (MLP), LeNet, CNN, LSTM

### Training

Script to run experiment is in `project/training/run_experiment.py`. In here, there is the option to use Weights & Biases (more about this further below). There is `GPUManager` class which allocates GPU resources. 

There is a Dataset class and Model class which can be extended by dataset-specific and model-specific classes respectively. 

Sample training code: `training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "train_args": {"batch_size": 256}}' --gpu=1`. To change accordingly depending on which dataset, model or network being used. 

`experiment_config` which is a dictionary can contain: 

`dataset_args` such as `subsample_function`, `max_length`, `max_overlap`

`network_args` such as `num_layers`, `dropout_amount`, `layer_size`, `window_width`, `window_stride`

`train_args` such as `batch_size`, `epochs`

### Weights & Biases (is also somewhat similar to polyaxon)

I used Weights & Biases to track my experiments and for hyperparameter tuning. Project page is [here](https://wandb.ai/desiree/handwriting-recognition-project_training?workspace=user-desiree). Experiments can be run individually (view the run titled absurd-microwave in my Weights & Biases page) or in multiples. Multiple experiments are defined in json file (`training/experiments/sample.json`).

Sweeps are used for hyperparameter tuning. Hyperparameter values are defined in `training/sweep_emnist.yaml` file. 
