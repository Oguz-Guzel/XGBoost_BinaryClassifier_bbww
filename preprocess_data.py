#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")
plt.rcParams["figure.dpi"] = 400

import json

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

pd.set_option("display.max_columns", None)

import argparse

parser = argparse.ArgumentParser(description="Data Preprocessing Script")
parser.add_argument(
    "--output_dir",
    type=str,
    help="Data output directory",
)
# parser.add_argument(
#     "--bamboo_results_dir",
#     type=str,
#     default="/eos/user/a/aguzel/bamboo-output/v1.3.0-2022-cf/results/",
#     help="Directory containing the Bamboo results",
# )
# parser.add_argument(
#     "--n_events",
#     type=int,
#     default=10_000,
#     help="Number of events to process (default: 1e4)",
# )
# parser.add_argument(
#     "--noise_level",
#     type=float,
#     default=0.,
#     help="Level of noise to add to the input features (default: 0, no noise)",
# )
args = parser.parse_args()

output_dir = args.output_dir
bamboo_results_dir = [
    "/eos/user/a/aguzel/bamboo-output/newMLvars-cutflow-2022/results/",
    "/eos/user/a/aguzel/bamboo-output/newMLvars-cutflow-2023/results/",
]
N = 1e10
N = int(N)  # Convert to integer for consistency
noise_level = 0.01


os.makedirs(f"{output_dir}", exist_ok=True)

batch_size = 1024

print(f"Using {output_dir} as output directory\n")
print(f"Batch size: {batch_size}\n")
if N > 1e6:
    print("Taking all events")
    print("(N =", N, ")\n")
else:
    print("Taking N =", N, " events\n")


def listFiles(prefixes):
    return [
        file
        for file in rootFiles
        if any(file.split("/")[-1].startswith(p) for p in prefixes)
    ]


# get the data files
rootFiles = []
for path in bamboo_results_dir:
    for file in os.listdir(path):
        rootFiles.append(os.path.join(path, file))

TT_files = listFiles(["TT"])
DY_files = listFiles(["DY"])
other_bkg_files = listFiles(["Tbar", "TW", "W", "Z"])
HH_files = listFiles(["ggH"])
VBF_files = listFiles(["VBF"])

config_dict = {
    "output_dir": output_dir,
    "bamboo_results_dir": bamboo_results_dir,
    "n_events": N,
    "noise_level": noise_level,
    "root_files": rootFiles,
    "TT_files": TT_files,
    "DY_files": DY_files,
    "other_bkg_files": other_bkg_files,
    "HH_files": HH_files,
}

with open(f"{output_dir}/config.json", "w") as f:
    json.dump(config_dict, f, indent=4)
print(f"Saved config to {output_dir}/config.json")

print("Number of TT root files:", len(TT_files))
print("Number of DY root files:", len(DY_files))
print("Number of other_bkg root files:", len(other_bkg_files))
print("Number of HH root files:", len(HH_files))
print("Number of VBF root files:", len(VBF_files))

treenames = [
    "DL_resolved_1b_ee_ml_vars;1",
    "DL_resolved_1b_mumu_ml_vars;1",
    "DL_resolved_1b_emu_ml_vars;1",
    "DL_resolved_2b_ee_ml_vars;1",
    "DL_resolved_2b_mumu_ml_vars;1",
    "DL_resolved_2b_emu_ml_vars;1",
    "DL_boosted_ee_ml_vars;1",
    "DL_boosted_mumu_ml_vars;1",
    "DL_boosted_emu_ml_vars;1",
]

# The variables for each object (here l1, l2, j1, j2, j3, j4, j8(ak8) and met)
input_features = {
    "lepton_1": [
        "l1_Px",
        "l1_Py",
        "l1_Pz",
        "l1_E",
        "l1_pdgId",
        "l1_charge",
        "leading_lepton_pt",
    ],
    "lepton_2": [
        "l2_Px",
        "l2_Py",
        "l2_Pz",
        "l2_E",
        "l2_pdgId",
        "l2_charge",
        "subleading_lepton_pt",
    ],
    "jet_1": ["j1_Px", "j1_Py", "j1_Pz", "j1_E", "j1_btag"],
    "jet_2": ["j2_Px", "j2_Py", "j2_Pz", "j2_E", "j2_btag"],
    "jet_3": ["j3_Px", "j3_Py", "j3_Pz", "j3_E", "j3_btag"],
    "jet_4": ["j4_Px", "j4_Py", "j4_Pz", "j4_E", "j4_btag"],
    "jet_8": [
        "j8_Px",
        "j8_Py",
        "j8_Pz",
        "j8_E",
        "j8_btag",
        "j8_tau1",
        "j8_tau2",
        "j8_tau3",
        "j8_tau4",
        "j8_msoftdrop",
    ],
    "met": ["met_Px", "met_Py", "met_E"],  # , "met_LD", "HT"], include these
    "misc": [
        "dR_l1_l2",
        "dR_j1_j2",
        "dR_dilepton_dijet",
        "dR_dilepton_dibjet",
        "abs_dphi_met_dilepton",
        "min_dR_l1_ak4jets",
        "min_dR_l2_ak4jets",
        "min_dR_lead_bjet_leptons",
        "min_dR_sublead_bjet_leptons",
        "min_dR_ak4jets",
        "min_abs_dphi_ak4jets",
        "di_bjet_mass",
        "di_lepton_mass",
        "di_lepton_met_mass",
        "VBF_tag",
        "boosted_tag",
        "run_year",
    ],
}

n_features = sum([len(val) for val in input_features.values()])

print("\nNumber of input features:", n_features)
print(f"which are: {[value for value in input_features.values()]}\n")

processes = ["HH", "bkg"]

target_set = [
    "HH",
]


import uproot
from tqdm import tqdm

# Collect all input feature names into a flat list
feature_list = [item for sublist in input_features.values() for item in sublist]


class CustomData:
    def __init__(
        self,
        files,
        type,
        treenames,
        feature_list,
        extra_cols=None,
        n_events=N,
        name="data",
    ):
        self.files = files
        self.type = type
        self.treenames = treenames
        self.feature_list = feature_list.copy()
        if extra_cols:
            self.feature_list += extra_cols
        self.extra_cols = extra_cols
        self.n_events = n_events
        self.name = name

        # Load the data from the files
        self.df = self.load_data()

        # Add one-hot encoded column based on type, and set as int32
        if self.type.lower() == "hh":
            self.df["HH"] = np.int8(1)
            self.df["bkg"] = np.int8(0)
        else:
            self.df["HH"] = np.int8(0)
            self.df["bkg"] = np.int8(1)

    def load_data(self):
        data_frames = []
        for file in tqdm(self.files, desc=f"Loading {self.name} data"):
            with uproot.open(file) as f:
                available_keys = f.keys()
                for treename in self.treenames:
                    # Remove cycle number for matching
                    base_treename = treename.split(";")[0]
                    matched_key = None
                    for key in available_keys:
                        if base_treename in key:
                            matched_key = key
                            break
                    if matched_key is not None:
                        tree = f[matched_key]
                        arrs = tree.arrays(
                            self.feature_list, entry_stop=self.n_events, library="pd"
                        )
                        data_frames.append(arrs)
                    else:
                        print(
                            f"Warning: Tree '{treename}' not found in {file.split('/')[-1]}. Skipping."
                        )
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if nothing found


extra_cols = ["event_no", "weight"]

DY_df = CustomData(
    DY_files,
    "bkg",
    treenames,
    feature_list,
    extra_cols=extra_cols,
    n_events=N,
    name="DY",
)

TT_df = CustomData(
    TT_files,
    "bkg",
    treenames,
    feature_list,
    extra_cols=extra_cols,
    n_events=N,
    name="TT",
)

other_bkg_df = CustomData(
    other_bkg_files,
    "bkg",
    treenames,
    feature_list,
    extra_cols=extra_cols,
    n_events=N,
    name="Other_bkg",
)

HH_df = CustomData(
    HH_files,
    "HH",
    treenames,
    feature_list,
    extra_cols=extra_cols,
    n_events=N,
    name="ggF_HH",
)


# Concatenate DataFrames from CustomData objects in customdata_set
customdata_set = [DY_df, TT_df, other_bkg_df, HH_df]
# customdata_set = [DY_df, HH_df]

for customdata in customdata_set:
    print(customdata.df.shape[0], " events in ", customdata.name)

dfs = [data.df for data in customdata_set]
df = pd.concat(dfs, ignore_index=True)
print(f"\nConcatenated DataFrame shape: {df.shape}")


for col, type in df.dtypes.items():
    print(f"Column: {col}, Type: {type}")


if "event_no" not in df.columns:
    raise KeyError("The DataFrame does not contain 'event_no' column.")

# Add target columns to the data in one-hot encoding
for process in processes:
    for data in customdata_set:
        if data.type == process:
            data.df[process] = np.ones_like(data.df["event_no"])
        else:
            data.df[process] = np.zeros_like(data.df["event_no"])

weight_branch = "weight"


# Make pandas dataframes out of the data
# if N is less than the number of events in the root files, then here
# we're falling into `sample` which shuffles the data since we want
# to take a really random subset. But this approach may improve bias,
# instead stratification should be done or all events should be used.
# df = pd.concat(
#     [data.get_df().sample(n=min(N, len(data.get_df()))) for data in data_set]
# )

# convert tags to integers
df["VBF_tag"] = df["VBF_tag"].astype(np.int8)
df["boosted_tag"] = df["boosted_tag"].astype(np.int8)

# # one-hot encode pdgId and charge of leptons in separate columns
# for lep in ["l1", "l2"]:
#     # pdgId one-hot as integers
#     pdgid_dummies = pd.get_dummies(df[f"{lep}_pdgId"], prefix=f"{lep}_pdgId").astype(int)
#     df = pd.concat([df, pdgid_dummies], axis=1)
#     # charge one-hot as integers
#     charge_dummies = pd.get_dummies(df[f"{lep}_charge"], prefix=f"{lep}_charge").astype(int)
#     df = pd.concat([df, charge_dummies], axis=1)


print("Number of input features after one-hot encoding:", n_features + 4 * 2)

assert df.shape[0] == len(
    df["event_no"]
), "Number of rows in the DataFrame does not match the number of event_no entries"


# data cleaning

# Check for infinite or nan values
# Show a boolean DataFrame where inf values are True

# df.drop(
#     columns=["sample", "file", "tree"], inplace=True
# )  # drop event_no column if exists
inf_mask = np.isinf(df.values)

# Get row and column indices of inf values
rows, cols = np.where(inf_mask)

for r, c in zip(rows, cols):
    print(f"Row: {df.index[r]}, Column: {df.columns[c]}, Value: {df.iloc[r, c]}")

# Find rows with inf values
rows_with_inf = np.where(inf_mask)[0]
rows_with_inf = np.unique(rows_with_inf)  # Unique row indices

# Drop those rows
df = df.drop(df.index[rows_with_inf]).reset_index(drop=True)
print(f"Dropped {len(rows_with_inf)} row(s) containing inf values.")


# Keep only rows where both leptons have allowed pdgId values
allowed_pdgids = {-11, 11, -13, 13}
mask = df["l1_pdgId"].isin(allowed_pdgids) & df["l2_pdgId"].isin(allowed_pdgids)
print(
    f"Dropping {len(df) - mask.sum()} events with lepton pdgId not in {allowed_pdgids}"
)
df = df[mask].reset_index(drop=True)

# Drop events where either lepton has a charge not equal to -1 or 1
allowed_charges = {-1, 1}
charge_mask = df["l1_charge"].isin(allowed_charges) & df["l2_charge"].isin(
    allowed_charges
)
print(
    f"Dropping {len(df) - charge_mask.sum()} events with lepton charge not in {allowed_charges}"
)
df = df[charge_mask].reset_index(drop=True)

from scipy.stats import zscore

numeric_df = df.select_dtypes(include=[np.number])
z_scores = np.abs(zscore(numeric_df, nan_policy="omit"))
threshold = 10
# experimentally z score of 10 gives around 1% of the outliers
outlier_mask = z_scores > threshold

outlier_rows = df[(outlier_mask).any(axis=1)]
print(f"Number of rows with outliers (z-score > {threshold}): {outlier_rows.shape[0]}")

# Drop outlier rows from the dataframe
df.drop(outlier_rows.index, inplace=True)
df.reset_index(drop=True, inplace=True)
print("Outliers dropped.")


# Need to cut out negative weights #
print(
    f"\nTotal weight sum = {df[weight_branch].sum():1.3e}, with {(df[weight_branch]<0).sum()} negative weight events"
)
df = df[df[weight_branch] > 0]
print(
    f"\nAfter cutting out negative weights : total weight sum = {df[weight_branch].sum():1.3e}"
)


# In case you restricted the number of events, need to rescale the weights
for process, data_process in zip(processes, customdata_set):
    ratio = (
        data_process.df[weight_branch].sum() / df[df[process] == 1][weight_branch].sum()
    )
    df.loc[df[process] == 1, weight_branch] *= ratio


# Plot the weights before normalisation
plt.hist(
    df[df["HH"] == 1][weight_branch],
    label="Signal (ggF HH)",
    histtype="step",
    linewidth=1,
    bins=100,
    color="blue",
)
plt.hist(
    df[df["bkg"] == 1][weight_branch],
    label="Background",
    histtype="step",
    linewidth=1,
    bins=100,
    color="red",
)
plt.xlabel("Event weight")
plt.xscale("log")
plt.yscale("log")
plt.title(
    "Event weights before normalisation",
    fontsize=24,
    loc="left",
    pad=10,
    fontproperties="Tex Gyre Heros:italic",
)
plt.title(
    "(13.6 TeV)", fontsize=24, loc="right", pad=10, fontproperties="Tex Gyre Heros"
)
plt.legend(loc="upper right")
plt.savefig(f"{output_dir}/event_weights.png")
plt.clf()
print(f"\nWeights saved to {output_dir}/event_weights.png")

# Now equalize the sum of weight
print("\nBefore reweighting")
for process in processes:
    print(
        f"{process} : N = {df[df[process]==1].shape[0]:6d}, sum(w) = {df[df[process]==1][weight_branch].sum():1.3e}"
    )

for process in processes:
    df.loc[df[process] == 1, weight_branch] *= (
        df.shape[0] / len(processes) / df[df[process] == 1][weight_branch].sum()
    )

print("After reweighting")
for process in processes:
    print(
        f"{process} : N = {df[df[process]==1].shape[0]:6d}, sum(w) = {df[df[process]==1][weight_branch].sum():1.3e}"
    )

# Increase the weight for HH events
df.loc[df["HH"] == 1, weight_branch] *= 5
print("\nAfter multiplying HH weights by 5")
for process in processes:
    print(
        f"{process} : N = {df[df[process]==1].shape[0]:6d}, sum(w) = {df[df[process]==1][weight_branch].sum():1.3e}"
    )

# Plot the weights after normalisation
plt.hist(
    df[df["HH"] == 1][weight_branch],
    label="Signal (ggF HH)",
    histtype="step",
    linewidth=1,
    bins=100,
    color="blue",
)
plt.hist(
    df[df["bkg"] == 1][weight_branch],
    label="Background",
    histtype="step",
    linewidth=1,
    bins=100,
    color="red",
)
plt.xlabel("Event weight")
plt.xscale("log")
plt.yscale("log")
plt.title(
    "Event weights after normalisation",
    fontsize=24,
    loc="left",
    pad=10,
    fontproperties="Tex Gyre Heros:italic",
)
plt.title(
    "(13.6 TeV)", fontsize=24, loc="right", pad=10, fontproperties="Tex Gyre Heros"
)
plt.legend(loc="upper right")
plt.savefig(f"{output_dir}/normalised_weights.png")
plt.clf()
print(f"\nNormalised weights saved to {output_dir}/normalised_weights.png")


# Convert all feature columns to float32 for efficiency
df[feature_list] = df[feature_list].astype(np.float32)
df[weight_branch] = df[weight_branch].astype(np.float32)
df["event_no"] = df["event_no"].astype(np.int64)


for col, type in df.dtypes.items():
    print(col, "    :", type)


# # Create directory for histograms if it doesn't exist
# hist_dir = f"{output_dir}/column_hists"
# os.makedirs(hist_dir, exist_ok=True)

# # Plot each column as a histogram with linear and log y-scale side by side
# for col in df.columns:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     # Linear scale
#     axes[0].hist(df[col], bins=50, color="skyblue")
#     axes[0].set_title(f"{col} (linear)")
#     axes[0].set_xlabel(col)
#     axes[0].set_ylabel("Count")
#     # Log scale
#     axes[1].hist(df[col], bins=50, color="skyblue")
#     axes[1].set_yscale("log")
#     axes[1].set_title(f"{col} (log)")
#     axes[1].set_xlabel(col)
#     axes[1].set_ylabel("Count")
#     plt.tight_layout()
#     fig.savefig(os.path.join(hist_dir, f"{col}_hist.png"))
#     plt.close(fig)
# print(f"Saved all inputs as histograms to {hist_dir}")


# if noise_level > 0.0:
#     # Inject some noise into the numeric columns
#     print("Adding noise.")
#     # List of columns to augment (exclude targets and weights)
#     exclude_cols = ["HH", "bkg", "weight"]

#     numeric_cols = [
#         col
#         for col in df.select_dtypes(include=[np.number]).columns
#         if col not in exclude_cols
#     ]

#     # Set noise which is noise_level % of each column's std

#     # Create augmented dataframe
#     df_aug = df.copy()
#     for col in numeric_cols:
#         std = df[col].std()
#         noise = np.random.normal(0, noise_level * std, size=len(df))
#         df_aug[col] = df[col] + noise

#     # Concatenate augmented data to original
#     df = pd.concat([df, df_aug], ignore_index=True)


df.to_parquet(f"{output_dir}/processed_data.parquet")
print(f"Processed DataFrame saved to {output_dir}/processed_data.parquet")
