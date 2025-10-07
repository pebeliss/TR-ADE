import tensorflow as tf
# Limit parallel threads for reproducibility
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import os, sys, json, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.keras
# import mlflow.tensorflow
# from mlflow.models import infer_signature

sys.path.append(".")
# from TR_ADE_pipeline import set_seed
import yaml
from pathlib import Path
from TR_ADE import *
# from TR_ADE_pipeline import PREPROCESSOR_WRAPPER, TR_ADE_WRAPPER
from TR_ADE_pipeline import *
from model_args import getArgs
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_curve, precision_recall_curve, auc, classification_report
)
import cloudpickle, joblib

# seed = 1000
# set_seed(seed)

def evaluate_and_log(probs, true_labels, clusters=None):
    y_true = true_labels.values.ravel()
    y_scores = probs[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_vals, precision_vals)

    mlflow.log_metric("f1", float(f1))
    mlflow.log_metric("roc_auc", float(roc_auc))
    mlflow.log_metric("pr_auc", float(pr_auc))
    # mlflow.log_param("seed", seed)

    # ROC figure
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"Overall ROC (AUC={roc_auc:.4f})")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.title("ROC")
    mlflow.log_figure(fig, "plots/roc.png")
    plt.close(fig)

    # PR figure
    fig = plt.figure()
    plt.plot(recall_vals, precision_vals, label=f"Overall PR (AUC={pr_auc:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.title("Precision-Recall")
    mlflow.log_figure(fig, "plots/pr.png")
    plt.close(fig)

    # Per-cluster optional
    if clusters is not None:
        for c in np.unique(clusters):
            m = clusters == c
            if m.sum() == 0: 
                continue
            f1_c = f1_score(y_true[m], (y_scores[m] >= 0.5).astype(int), zero_division=0)
            fpr_c, tpr_c, _ = roc_curve(y_true[m], y_scores[m])
            roc_auc_c = auc(fpr_c, tpr_c)
            p_c, r_c, _ = precision_recall_curve(y_true[m], y_scores[m])
            pr_auc_c = auc(r_c, p_c)
            mlflow.log_metric(f"f1_cluster_{int(c)}", float(f1_c))
            mlflow.log_metric(f"roc_auc_cluster_{int(c)}", float(roc_auc_c))
            mlflow.log_metric(f"pr_auc_cluster_{int(c)}", float(pr_auc_c))

def main():
    args = getArgs(sys.argv[1:])  # use CLI args
    # os.environ["PYTHONHASHSEED"] = str(args.seed) if hasattr(args, "seed") else "0"
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "train.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        args.__dict__.update({k: v for k, v in cfg.items() if k in args.__dict__})
    set_seed(args.seed)
    
    # Start MLflow run
    mlflow.set_experiment("vae_classifier")
    with mlflow.start_run():

        # Load data (your current notebook workflow)
        heartdata = pd.read_csv(os.path.join(args.data_path, "heart.csv"))
        gender = {'M': 0, 'F': 1}
        angina = {'N': 0, 'Y': 1}
        heartdata["Sex"] = [gender[i] for i in heartdata["Sex"]]
        heartdata["ExerciseAngina"] = [angina[i] for i in heartdata["ExerciseAngina"]]

        y_data = heartdata["HeartDisease"]
        X_data = heartdata[[c for c in heartdata.columns if c != "HeartDisease"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data.values.reshape(-1,1), stratify=y_data, random_state=42
        )
        args.num_classes = len(np.unique(y_train))
        
        # log all argparse params
        mlflow.log_params({k: v for k, v in vars(args).items()})

        # preprocess & generators
        preprocessor = PREPROCESSOR_WRAPPER(args)
        preprocessor.build(X_train, X_test)
        train_gen, val_gen, train_data, val_data = preprocessor.data_pipeline(
            X_train, y_train, return_generator=True
        )

        # build & train
        tr_ade = TR_ADE_WRAPPER(args, preprocessor.cont_dim, preprocessor.bin_dim)
        tr_ade.build()

        # Optionally add a simple MLflow Keras callback (logs train/val metrics)
        mlf_cb = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: mlflow.log_metrics(
                {f"epoch_{k}": float(v) for k, v in logs.items() if isinstance(v, (int,float))},
                step=epoch
            )
        )
        print(tr_ade.TR_ADE.c_mu)
        
        history = tr_ade.model_fit(train_gen, val_gen, callbacks=[mlf_cb])  # slight change; see §2.2

        # Log final losses from history if present
        if hasattr(history, "history"):
            for k, v in history.history.items():
                if len(v) > 0:
                    mlflow.log_metric(f"final_{k}", float(v[-1]))

        # Latent visualization (log as artifact)
        X_all = pd.concat([train_data[0], val_data[0]], axis=0).loc[X_train.index].reset_index(drop=True)
        y_tr = train_data[2]; y_tr.index = train_data[0].index
        y_vl = val_data[2];  y_vl.index = val_data[0].index
        y_all = pd.concat([y_tr, y_vl], axis=0).loc[X_train.index].reset_index(drop=True)

        z_mean, z_logvar, z = tr_ade.TR_ADE.encoder(X_all)
        clusters = tr_ade.TR_ADE.get_clusters(z_mean)
        x_mean_pred, x_logvar_pred, theta_pred = tr_ade.TR_ADE.decoder(z_mean)
        reconstructed = tf.concat([x_mean_pred, theta_pred], axis=1)
        
        print(np.unique(clusters, return_counts=True))
        
        fig = plt.figure()
        plt.scatter(z_mean[:,0], z_mean[:,1], c=clusters)
        plt.colorbar(); plt.title("Latent space (clusters)")
        mlflow.log_figure(fig, "plots/latent_clusters.png")
        plt.close(fig)
        
        fig = plt.figure()
        palette = sns.color_palette("tab10", len(np.unique(clusters)))
        colors = {cluster: palette[i] for i, cluster in enumerate(np.unique(clusters))}

        for i in range(args.num_clusters):
            if i in np.unique(clusters):
                cluster_samples = np.random.multivariate_normal(tr_ade.TR_ADE.c_mu[i,:].numpy(), np.diag(tr_ade.TR_ADE.log_c_sigma[i,:]), size = 100)
                plt.scatter(cluster_samples[:,0],cluster_samples[:,1], label = f'Cluster {i}', s=5, color=colors[i])
                sns.kdeplot(x=cluster_samples[:,0], y=cluster_samples[:,1], linewidths=0.5, color=colors[i])        
        plt.scatter(z_mean[:,0], z_mean[:,1], color='black', label='Z', s=5)
        plt.scatter(z_mean[:,0], z_mean[:,1], c=clusters, s=3, cmap='cool')
        plt.colorbar()
        plt.legend()
        mlflow.log_figure(fig, "plots/latent_and_gen_samples.png")
        plt.close(fig)

        # Test set inference
        test_data = preprocessor.preprocess(X_test, y_test)
        X_recon, z_mean_te, clusters_te, pred_label = tr_ade.model_predict(test_data[0])

        # Evaluation (logs metrics + figs)
        evaluate_and_log(pred_label, test_data[2].iloc[:,1], clusters=clusters_te)

        # Save & log artifacts: preprocessor and model
        os.makedirs("artifacts", exist_ok=True)
        preproc_path = "artifacts/preprocessor.pkl"
        with open(preproc_path, "wb") as f:
            cloudpickle.dump(preprocessor, f)
        mlflow.log_artifact(preproc_path)

        # Log Keras model (use TF flavor)
        # mlflow.tensorflow.log_model(
        #     tf_saved_model_dir=tr_ade.TR_ADE,  # Keras/TF model object
        #     artifact_path="tr_ade",
        #     registered_model_name="VAEClassifier"
        # )
        try:
            # Newer MLflow versions usually support extra kwargs like registered_model_name
            mlflow.keras.log_model(
                tr_ade.TR_ADE,                      # <-- positional, NOT keras_model=
                name="model",
                registered_model_name="VAEClassifier",
                keras_module="tensorflow.keras",  # remove if this triggers TypeError
                pip_requirements = args.pip_reqs_path
            )
        except TypeError:
            # Older MLflow: fewer kwargs; fall back to simplest signature
            with open(args.pip_reqs_path) as f:
                req_list = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
            mlflow.keras.log_model(tr_ade.TR_ADE, "model", pip_requirements=req_list)
                # Optional: tag the run
        mlflow.set_tag("framework", "tf-keras")
        mlflow.set_tag("pipeline", "TR_ADE_pipeline.py")

if __name__ == "__main__":
    main()
