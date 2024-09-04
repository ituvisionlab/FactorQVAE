"""
InfoMEC implementation is adapted from its original Tensorflow implementation.
Source: https://github.com/kylehkhsu/latent_quantization/blob/main/disentangle/metrics/infomec.py
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
from sklearn import preprocessing, feature_selection, metrics, linear_model
import matplotlib.pyplot as plt 

def compute_infomec(data, model, representation_function, random_state, num_train, batch_size=16):
    scores = {}
    mus_train, ys_train = generate_batch_factor_code(data, model, representation_function, num_train, random_state, batch_size)
    mus_train = mus_train.T
    ys_train = ys_train.T
    assert mus_train.shape[0] == num_train
    assert ys_train.shape[0] == num_train
    print("Generated batch factor code")
    if model.config.model.name in ["QLAE", "DVAE", "FactorDVAE", "QVAE", "FactorQVAE"]:
        is_discrete = True
    else:
        is_discrete = False
    
    nmi = compute_nmi(mus_train, ys_train, is_discrete, random_state)
    print("Computed NMI")

    latent_ranges = np.max(mus_train, axis=0) - np.min(mus_train, axis=0)

    if is_discrete:
        active_latents = latent_ranges > 0
    else:
        active_latents = latent_ranges > np.max(latent_ranges) / 20

    num_sources = ys_train.shape[1]
    num_active_latents = np.sum(active_latents)
    pruned_nmi = nmi[:, active_latents]

    infom = (np.mean(np.max(pruned_nmi, axis=0) / np.sum(pruned_nmi, axis=0)) - 1 / num_sources) / (1 - 1 / num_sources)
    infoc = (np.mean(np.max(pruned_nmi, axis=1) / np.sum(pruned_nmi, axis=1)) - 1 / num_active_latents) / (1 - 1 / num_active_latents)
    infoe = compute_infoe(mus_train, ys_train, is_discrete, random_state)

    scores["infom"] = infom
    scores["infoe"] = infoe
    scores["infoc"] = infoc
    scores["active_latents"] = active_latents

    return nmi, scores

def compute_nmi(mus, ys, is_discrete, random_state):
    processed_ys = process_ys(ys)
    processed_mus = []

    if is_discrete:
        processed_mus = mus
    else:
        for j in range(mus.shape[1]):
            processed_mus.append(preprocessing.StandardScaler().fit_transform(mus[:, j][:, None]))
        processed_mus = np.concatenate(processed_mus, axis=1)
    
    ret = np.empty(shape=(processed_ys.shape[1], processed_mus.shape[1]))

    for i in range(processed_ys.shape[1]):
        for j in range(processed_mus.shape[1]):
            if is_discrete:
                ret[i, j] = metrics.mutual_info_score(processed_ys[:, i], processed_mus[:, j])
            else:
                ret[i, j] = feature_selection.mutual_info_classif(processed_mus[:, j][:, None], processed_ys[:, i],
                                                                random_state=random_state)
        entropy = metrics.mutual_info_score(processed_ys[:, i], processed_ys[:, i])
        ret[i, :] = ret[i, :] / entropy
    
    return ret

def process_ys(ys):
    processed_ys = []
    for i in range(ys.shape[1]):
        processed_ys.append(preprocessing.LabelEncoder().fit_transform(ys[:, i]))
    processed_ys = np.stack(processed_ys, axis=1)

    return processed_ys

def compute_infoe(mus, ys, is_discrete, random_state):
    normalized_predictive_information_per_source = []
    processed_ys = process_ys(ys)

    if is_discrete:
        processed_mus = preprocessing.OneHotEncoder(sparse_output=False).fit_transform(mus)
    else:
        processed_mus = preprocessing.StandardScaler().fit_transform(mus)
    
    for i in range(processed_ys.shape[1]):
        predictive_conditional_entropy = logistic_regression(processed_mus, processed_ys[:, i], random_state)
        null = np.zeros_like(mus)
        marginal_ys_entropy = logistic_regression(null, processed_ys[:, i], random_state)

        normalized_predictive_information_per_source.append((marginal_ys_entropy - predictive_conditional_entropy) / marginal_ys_entropy)
    
    return np.mean(normalized_predictive_information_per_source)

def logistic_regression(X, y, random_state):
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1

    model = linear_model.LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=500,
        multi_class='multinomial',
        n_jobs=-1,
        random_state=random_state
    )

    model.fit(X, y)
    y_pred = model.predict_proba(X)

    return metrics.log_loss(y, y_pred)

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="InfoMEC")
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

    nmi_matrix, scores = compute_infomec(data_module.dataset, model, quantizer_representation_function, np.random.RandomState(config.train.seed), args.num_train, args.batch_size)
    print(scores)
    with open(os.path.join(exp_path, "nmi_matrix.npy"), "wb") as f:
        np.save(f, nmi_matrix)       

    with open(os.path.join(exp_path, "infomec.txt"), "w") as f:
        print(scores, file=f)
