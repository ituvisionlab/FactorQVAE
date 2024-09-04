"""
DCI implementation is adapted from its Tensorflow implementation.
Source: https://github.com/google-research/disentanglement_lib/tree/master
"""
import os
import glob
import sys
sys.path.append('..')

from metrics.utils import *

from data.shapes3d import Shapes3DDataModule
from data.isaac3d import Isaac3DDataModule
from data.mpi3d_complex import MPI3DComplexDataModule

from models.ae import AE
from models.bioae import BioAE
from models.dvae import DVAE
from models.factor_dvae import FactorDVAE
from models.factor_qvae import FactorQVAE
from models.factor_vae import FactorVAE
from models.qlae import QLAE
from models.qvae import QVAE
from models.vae import VAE

import numpy as np
import scipy
from sklearn import ensemble
import matplotlib.pyplot as plt 

def compute_dci(data, model, representation_function, random_state, num_train, num_test, batch_size=16):
    mus_train, ys_train = generate_batch_factor_code(data, model, representation_function, num_train, random_state, batch_size)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    print("Generated batch factor code")
    mus_test, ys_test = generate_batch_factor_code(
        data, model, representation_function, num_test,
        random_state, batch_size)
    importance_matrix, scores = _compute_dci(mus_train, ys_train, mus_test, ys_test)

    return importance_matrix, scores

def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    print("Computing DCI...")
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return importance_matrix, scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    print("Computing Importance Matrix")
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        print("Training Regressor {}".format(i))
        model = ensemble.GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    print("Computing Disentanglement")
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    print("Computing Completeness")
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor*factor_importance)

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="DCI")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantizer", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    exp_path = os.path.join("..", args.model, args.quantizer, 
                            args.data, args.exp_name)
    config_path = glob.glob(os.path.join(exp_path, "codes", "*.yaml"))[0]
    ckpt_path = os.path.join(exp_path, "ckpt", args.ckpt)

    config = load_config(config_path)
    config.data.data_path = "." + config.data.data_path

    data_module = eval(config.data.name)(config)
    data_module.setup()

    model = eval(config.model.name)(config, img_dir=None).cuda()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    importance_matrix, scores = compute_dci(data_module.dataset, model, quantizer_representation_function, np.random.RandomState(config.train.seed), args.num_train, args.num_test, args.batch_size)
    print(scores)

    with open(os.path.join(exp_path, "importance_matrix.npy"), "wb") as f:
        np.save(f, importance_matrix)     

    with open(os.path.join(exp_path, "dci.txt"), "w") as f:
        print(scores, file=f)
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")
            print(f"{k}: {v:.4f}", file=f)

    
        

