# NITO: Neural Implicit Fields for Resolution-free and Domain-Adaptable Topology Optimization
This repo serves as the official code base for NITO.
![NITO_2](https://github.com/user-attachments/assets/c82fe4df-9f54-481d-a165-87192249ac83)
In this repo we include the code used to perform the experiments on NITO. Below is a detailed explaniation of the code provided.

# Data & Checkpoints
To download data you can run:

```bash
python download.py --data
```

Similarly to download checkpoints:

```bash
python download.py --checkpoints
```

You can download both data and checkpoints by running:

```bash
python download.py --all
```

Note: Since `download.py` uses `gdown` to download files, google drive may limit the download quota. If you encounter this issue, you can download the files manually from the following link:

[Data & Checkpoints](https://drive.google.com/drive/folders/1_wKPq8HXjaoRa4oCy_tvLOopIcapk7wO?usp=sharing)

[3D Data](https://drive.google.com/drive/folders/1uK_X3-FcCWY9LiiXkVQDI69q0t6Vosgm)

# Replicating Experiments
The raw code for NITO model is provided in the NITO module in `/NITO` folder. The code for training and evaluation is provided in the `train.py` and `evaluate.py` files respectively, which have arguments that can be found with more details by running `--help` flag. The code for our optimized SIMP optimizer is provided in the ATOMS module in `/ATOMS` folder. Most of this code well commented and self explanatory but the `evaluate.py` file includes code to run this optimizer which maybe helpful to understand the code.

## Replicating NITO Experiments From Scratch
To replicate the experiments from scratch, you can run the following script to train all model variants (ensure data is downloaded):

```bash
bash run_train_experiments.sh
```

Then you can run the following script to evaluate the models:

```bash
bash run_evaluation_experiments.sh
```

This will train all the models and evaluate them. The results will be saved in the `/Results` folder and model chekpoints can be found in the `/Checkpoints` folder.

## Replicating NITO Experiments Using Pretrained Models
To replicate the experiments using pretrained models, you can run the following script to evaluate the models (ensure data and checkpoints are downloaded):

```bash
bash run_evaluation_experiments.sh
```

# Environment Setup
Please use `environment.yml` to create a conda environment with all the dependencies.
