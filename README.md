# FactorQVAE
Official implementation of "Disentanglement with Factor Quantized Variational Autoencoders"

## Installation
```
conda env create -f environment.yml
conda activate factorqvae
```
## Data Preparation
Run the files in `scripts` folder:
```
python prepare_shapes3d.py
python prepare_isaac3d.py
python prepare_mpi3dcomplex.py
```
All the datasets will be downloaded to `datasets/{dataset_name}` folders.
## Repository Configuration

`blocks`: Consists of building blocks of the architecture.

`configs/{dataset_name}`: Consists of model specific configuration files for each dataset.

`data`: Consists of data loader files.

`decoders`: Consists of decoder architectures for each model.

`encoders`: Consists of encoder architectures for each model.

`metrics`: Consists of disentanglement metric implementations.

`models`: Consists of all model implementations.

`quantizers`: Consists of quantization/sampling operations.

`histogram.py`: Codebook usage visualization for discrete VAE variants.

`latent_traversal.py`: Performing latent manipulation and reconstruction.

`train.py:` Training file.

`utils.py`: Helper functions.

## Training

In order to train a model, run the following command:
```
python train.py --config_path configs/{dataset_name}/{model_name}.py
```
Example:
```
python train.py --config_path configs/Isaac3D/factorqvae.py
```
Each experiment's result be saved in a folder `{model_name}/{quantizer_name}/{lightning_data_module_name}/{experiment_date}`, e.g. `FactorQVAE/RelaxedVQQuantizer/Isaac3DDataModule/2024-09-03_17-18-38`. 
Each folder will consist of `ckpt`, `codes`, `imgs`, and `logs` folders. Checkpoints will be saved in `ckpt`, codes that are used in this experiment will be saved in `codes`, reconstruction results will be saved in `imgs`, and tensorboard logs will be saved in `logs`.

## Evaluation

In order to evaluate a model in terms of disentanglement, go to `metrics` folder, and run:
```
python dci.py --model {model_name} --quantizer {quantizer_name} --data {lightning_data_module_name} --exp_name {experiment_date} --ckpt {ckpt_name}
python infomec.py --model {model_name} --quantizer {quantizer_name} --data {lightning_data_module_name} --exp_name {experiment_date} --ckpt {ckpt_name}
```
Example:
```
python dci.py --model FactorQVAE --quantizer RelaxedVQQuantizer --data Isaac3DDataModule --exp_name 2024-09-03_17-18-38 --ckpt epoch=22-step=100000.ckpt
python infomec.py --model FactorQVAE --quantizer RelaxedVQQuantizer --data Isaac3DDataModule --exp_name 2024-09-03_17-18-38 --ckpt epoch=22-step=100000.ckpt
```
After running these commands, txt files (`dci.tx`, `infomec.txt`) including the evaluation scores will be saved in the experiment folder. Moreover, `importance_matrix.npy` and `nmi_matrix.npy` used for DCI and InfoMEC, respectively, will be saved in the experiment folder.

For visual latent traversal, run:
```
python latent_traversal.py --model {model_name} --quantizer {quantizer_name} --data {lightning_data_module_name} --exp_name {experiment_date} --ckpt {ckpt_name}
```
This will create a new folder `latent_traversal` in the experiment folder, and 5 randomly sampled images will be intervened on each latent variable.

For codebook usage visualization, run:
```
python histogram.py --model {model_name} --quantizer {quantizer_name} --data {lightning_data_module_name} --exp_name {experiment_date} --ckpt {ckpt_name}
```
This will create a new folder `hist_for_each_position` in the experiment folder, and html files for each latent variable showing which codebook elements are used to quantize that latent variable with their frequencies will be saved. 

## Acknowledgement

This codebase is prepared considering the official/highly used implementations of [DCI](https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/dci.py), [InfoMEC](https://github.com/kylehkhsu/latent_quantization/blob/main/disentangle/metrics/infomec.py), and [dVAE](https://github.com/karpathy/deep-vector-quantization/tree/main).
