from transformers import AutoModel, AutoConfig
from  carbontracker.tracker import CarbonTracker
import json, torch, time
from transformers import AutoModel
import pandas as pd
import torch
import numpy as np
import requests

# URL of the ETTh1.csv dataset from the official ETT repository
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
output_file = "etth.csv"

def download_etth1(url, output_file):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Downloaded successfully: {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download: {e}")

# Get ETTH dataset
download_etth1(url, output_file)


# Load the model configuration
model_name = "ibm-research/testing-patchtst_etth1_pretrain"
config = AutoConfig.from_pretrained(model_name)

# Load the pretrained model
model = AutoModel.from_pretrained(model_name, config=config)

# Set to evaluation mode
model.eval()
print("Model loaded successfully!")


# Load the ETTh1 dataset
df = pd.read_csv("etth.csv")

# Display first few rows
print(df.head())

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
input_sequences = torch.stack([normalized_data[i:i+seq_length] for i in range(num_samples)]).to("cuda")

print("Input shape:", input_sequences.shape)  


#Load Model
model_name = "ibm-granite/granite-timeseries-patchtst"
model = AutoModel.from_pretrained(model_name)

#Set model to CUDA
model = model.to("cuda")

# Get Model Number of Params
num_params = sum(p.numel() for p in model.parameters())
print("Params", num_params)

energies = []
# Testing Multiple times
for _ in range(10):
    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Wait 5 seconds 
    time.sleep(5)

    # Initialize CarbonTracker
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()

    # Inference
    try:
        with torch.no_grad():
            outputs = model(input_sequences)
    except:
        raise(0)

    # Capture energy
    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]
    
    # Saving information
    energies.append({"tim": timing, "energy":energy, "divided":divided})
info = {'num_params':num_params,'energies':energies}
print(info)