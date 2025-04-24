from transformers import AutoModel, AutoConfig
from  carbontracker.tracker import CarbonTracker
import json, torch, time

# Load the model configuration
model_name = "ibm-research/testing-patchtst_etth1_pretrain"
config = AutoConfig.from_pretrained(model_name)

# Load the pretrained model
model = AutoModel.from_pretrained(model_name, config=config)

# Set to evaluation mode
model.eval()

print("Model loaded successfully!")

import pandas as pd

# Load the ETTh1 dataset
df = pd.read_csv("etth.csv")

# Display first few rows
print(df.head())

import torch
import numpy as np

# Drop the date column
df_numeric = df.drop(columns=["date"])

# Convert to torch tensor
time_series_data = torch.tensor(df_numeric.values, dtype=torch.float32)

# Normalize the data (zero mean, unit variance)
mean = time_series_data.mean(dim=0)
std = time_series_data.std(dim=0)
normalized_data = (time_series_data - mean) / std

print("Processed Data Shape:", normalized_data.shape)

seq_length = 512
num_samples = normalized_data.shape[0] - seq_length

# Create input sequences
input_sequences = torch.stack([normalized_data[i:i+seq_length] for i in range(num_samples)])

print("Input shape:", input_sequences.shape)  # Expected: (num_samples, seq_length, num_features)

from transformers import AutoModel

# Load the pretrained PatchTST model
#model_name = "ibm-research/testing-patchtst_etth1_pretrain"
#model_name = "ibm-research/patchtsmixer-etth1-pretrain"
model_name = "ibm-granite/granite-timeseries-patchtst"
model = AutoModel.from_pretrained(model_name)

num_params = sum(p.numel() for p in model.parameters())
print("Params", num_params)

num_params = sum(p.numel() for p in model.parameters())
energies = []
for _ in range(10):
    torch.cuda.empty_cache()
    time.sleep(5)
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()
    try:
        with torch.no_grad():
            outputs = model(input_sequences)
    except:
        raise(0)

    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]
    energies.append({"tim": timing, "energy":energy, "divided":divided})
info = {'num_params':num_params,'energies':energies}
print(info)