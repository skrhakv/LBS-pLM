# Fine-tuning protein language model for detection of protein-ligand binding sites
This repository contains code for fine-tuning & evaluating the ESM2-650M protein language model (pLM). The model was trained on the human subset of LIGYSIS dataset. For selecting the parameters, a random validation subset was selected. The repository loosely follows the fine-tuning pipeline from [this study](https://dl.acm.org/doi/10.1145/3765612.3767221).

## Repository structure
The repository contains:
1. `train.py`, `run-sbatch.sh`: scripts for training the ESM2-650M pLM on the SLURM cluster
2. `predict.ipynb`: a demonstration how the model can be used on GPU-equipped machine
3. `data`: training data

## External libraries
The pipeline uses libraries from [another repository](https://github.com/skrhakv/cryptic-finetuning), namely:
1. [finetuning_utils.py](https://github.com/skrhakv/cryptic-finetuning/blob/master/src/finetuning_utils.py)
2. [baseline_utils.py](https://github.com/skrhakv/cryptic-finetuning/blob/master/src/baseline_utils.py)

## Setup
To run the prediction, or to replicate the training, you need to create virtual environment and install appropriate libraries specified in `requirements.txt`

## Contact
If you have any questions, don't hesitate to raise an issue!

## License
This code is licensed under the [MIT license](https://github.com/skrhakv/LBS-pLM/blob/master/LICENSE).