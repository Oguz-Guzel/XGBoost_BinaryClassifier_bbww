import os
import json
import argparse
from time import time

parser = argparse.ArgumentParser(description="Transformer Training Script")
parser.add_argument(
    "--input_parquet",
    type=str,
    help="Path to the input parquet file containing preprocessed data",
    default="./preprocessed_data/processed_data.parquet",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Results directory",
)
parser.add_argument(
    "--n_estimators",
    type=int,
    default=2000,
    help="Number of trees in the XGBoost model",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.005,
    help="Learning rate for the XGBoost model",
)
parser.add_argument(
    "--max_depth",
    type=int,
    default=6,
    help="Maximum depth of the trees in the XGBoost model",
)

args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

n_estimators = args.n_estimators
learning_rate = args.learning_rate
max_depth = args.max_depth

seed = 42

config_dict = {
    "output_dir": output_dir,
    "seed": seed,
    "n_estimators": n_estimators,
    "learning_rate": learning_rate,
    "max_depth": max_depth,
}

with open(f"{output_dir}/config.json", "w") as f:
    json.dump(config_dict, f, indent=4)
print(f"Saved config to {output_dir}/config.json")

import pandas as pd

start_time = time()
df = pd.read_parquet("./preprocessed_data/processed_data.parquet")
print(f"Loaded data with {len(df)} samples and {len(df.columns)} features")
print(f"Data loading took {time() - start_time:.2f} seconds")

import xgboost as xgb

from sklearn.metrics import roc_auc_score, roc_curve

# Prepare features and target
exclude_cols = ["HH", "bkg", "weight", "event_no"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values
y = df["HH"].values
weights = df["weight"].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=seed, stratify=y
)


# Train an XGBoost model

model = xgb.XGBClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="gpu",
    random_state=seed,
    eval_metric=["logloss", "auc"],
    reg_alpha=0.1,
    reg_lambda=1.0,
)
model.fit(
    X_train,
    y_train,
    sample_weight=weights_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=1,
)


# obtain predictions
y_pred = model.predict_proba(X_test)[:, 1]
results = model.evals_result()

import matplotlib.pyplot as plt

# Predict and evaluate
roc_auc = roc_auc_score(y_test, y_pred, sample_weight=weights_test)
print(f"XGBoost ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred, sample_weight=weights_test)
plt.figure()
plt.plot(fpr, tpr, label=f"XGBoost ROC (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/xgb_roc_curve.png")
plt.show()
print(f"ROC curve saved to {output_dir}/xgb_roc_curve.png")


plt.figure()
plt.plot(
    results["validation_0"]["logloss"],
    label="Training Logloss",
    color="blue",
    marker="o",
    markersize=4,
    alpha=0.5,
)
plt.plot(
    results["validation_1"]["logloss"],
    label="Validation Logloss",
    color="red",
    marker="s",
    markersize=4,
    alpha=0.5,
)
plt.xlabel("Boosting Round")
plt.ylabel("Logloss")
plt.title("XGBoost Training and Validation Loss")
plt.legend()
plt.yscale("log")
plt.grid(True)
plt.savefig(f"{output_dir}/xgb_loss_curve.png")
plt.show()


# Plot score distribution for signal and background (normalised)
plt.figure()
plt.hist(
    y_pred[y_test == 1],
    weights=weights_test[y_test == 1],
    bins=50,
    histtype="step",
    color="blue",
    label="Signal",
    density=True,  # Normalise histogram
)
plt.hist(
    y_pred[y_test == 0],
    weights=weights_test[y_test == 0],
    bins=50,
    histtype="step",
    color="red",
    label="Background",
    density=True,  # Normalise histogram
)
plt.xlabel("XGBoost Output Score")
plt.ylabel("Density")
plt.legend()
plt.title("XGBoost Score Distribution (Normalised)")
plt.savefig(f"{output_dir}/xgb_score_dist_normalised.png")
plt.show()
print(f"Score distribution saved to {output_dir}/xgb_score_dist_normalised.png")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict class labels for validation set
y_pred_label = (y_pred > 0.5).astype(int)

# Compute confusion matrix (normalized)
cm = confusion_matrix(
    y_test, y_pred_label, sample_weight=weights_test, normalize="true"
)
display_labels = ["Background", "Signal"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap="Blues", values_format=".2f")
disp.ax_.set_yticklabels(display_labels, rotation=90)

disp.ax_.text(
    0.0,
    1.05,
    "Private work (CMS simulation)",
    fontsize=20,
    fontproperties="Tex Gyre Heros:italic",
    transform=disp.ax_.transAxes,
    verticalalignment="top",
)
disp.ax_.text(
    0.8,
    1.05,
    "(13.6 TeV)",
    fontsize=20,
    fontproperties="Tex Gyre Heros",
    transform=disp.ax_.transAxes,
    verticalalignment="top",
)

# Adjust colorbar to have the same height as the confusion matrix plot
cbar = disp.figure_.axes[-1]
cbar.set_position(
    [
        cbar.get_position().x0,
        disp.ax_.get_position().y0,
        cbar.get_position().width,
        disp.ax_.get_position().height,
    ]
)
plt.savefig(f"{output_dir}/xgb_confusion_matrix.png")
plt.show()
print(f"Confusion matrix saved to {output_dir}/xgb_confusion_matrix.png")


from xgboost import plot_importance

# Plot feature importance with feature names
plt.figure(figsize=(10, 6))
ax = plot_importance(
    model,
    max_num_features=len(feature_cols),
    importance_type="gain",
    show_values=False,
)
feature_names = [col for col in df.columns if col not in ["HH", "bkg", "weight"]]
ax.set_yticklabels([feature_names[i] for i in range(len(ax.get_yticklabels()))])
plt.title("XGBoost Feature Importance")
plt.tight_layout()
ax.tick_params(axis="y", labelsize=8, length=0)  # Remove y-axis ticks
plt.savefig(f"{output_dir}/xgb_feature_importance.png")
plt.show()
print(f"Feature importance plot saved to {output_dir}/xgb_feature_importance.png")


# from sklearn.model_selection import StratifiedKFold

# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
#     print(f"Fold {fold+1}")
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#     w_train, weights_test = weights[train_idx], weights[test_idx]

#     model.fit(
#         X_train,
#         y_train,
#         sample_weight=w_train,
#         eval_set=[(X_test, y_test)],
#         sample_weight_eval_set=[weights_test],
#         verbose=False,
#     )
#     y_pred = model.predict_proba(X_test)[:, 1]
#     roc_auc = roc_auc_score(y_test, y_pred, sample_weight=weights_test)
#     print(f"ROC AUC (fold {fold+1}): {roc_auc:.4f}")


import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

onnx_model = onnxmltools.convert_xgboost(
    model,
    initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))],
)
onnx_path = f"{output_dir}/model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"XGBoost model exported to {onnx_path}")


# import onnxruntime as ort
# import numpy as np

# sess = ort.InferenceSession(onnx_path)
# input_name = sess.get_inputs()[0].name
# output_names = [o.name for o in sess.get_outputs()]

# # Example input: use a real sample from your training data
# signal_sample = df[df.HH == 1].sample(n=1000, random_state=seed)[feature_cols].values
# bkg_sample = df[df.bkg == 1].sample(n=1000, random_state=seed)[feature_cols].values

# signal_outputs = sess.run(output_names, {input_name: signal_sample.astype(np.float32)})
# bkg_outputs = sess.run(output_names, {input_name: bkg_sample.astype(np.float32)})
# signal_probs = signal_outputs[1]
# bkg_probs = bkg_outputs[1]


# plt.figure()
# plt.hist(
#     signal_probs[:, 1],
#     bins=50,
#     color="skyblue",
#     histtype="step",
#     edgecolor="navy",
#     label="signal",
# )
# plt.hist(
#     bkg_probs[:, 1],
#     bins=50,
#     color="salmon",
#     histtype="step",
#     edgecolor="red",
#     label="background",
# )
# plt.xlabel("Predicted Probability (class 1)")
# plt.ylabel("Count")
# plt.title("ONNX Model Output Probabilities")
# plt.grid(True)
# plt.legend()
# plt.savefig(f"{output_dir}/onnx_model_output_probs.png")
# print(f"ONNX model output probabilities saved to {output_dir}/onnx_model_output_probs.png")


# import nbformat
# from nbconvert import PythonExporter

# notebook_path = "XGBoost.ipynb"
# script_path = "XGBoost.py"

# with open(notebook_path, "r", encoding="utf-8") as f:
#     nb = nbformat.read(f, as_version=4)

# python_exporter = PythonExporter()
# script_body, _ = python_exporter.from_notebook_node(nb)

# with open(script_path, "w", encoding="utf-8") as f:
#     f.write(script_body)

# print(f"Notebook converted to Python script: {script_path}")
