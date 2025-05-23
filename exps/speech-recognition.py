from urllib.request import urlopen
from PIL import Image
import timm
import time
import json
import torch
from  carbontracker.tracker import CarbonTracker
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCTC


from datasets import load_dataset
import pandas as pd
import os
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# Load the dataset
librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

# Take a sample of 50 examples
sampled_data = librispeech_test_clean.take(50)

# Convert the dataset to a list to work with it
sampled_data = list(sampled_data)
for i,d in enumerate(sampled_data):
    sampled_data[i]['audio']['raw'] = d['audio']['array']

#Define Model
data_models = ["openai/whisper-tiny.en"]

for model_name in data_models:
    #Load Model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device="cuda",
    )
    num_params = sum(p.numel() for p in model.parameters())
    energies = []
    #Testing multiple times
    for _ in range(10):
        # Convert the dataset to a list to work with it
        sampled_data = librispeech_test_clean.take(50)
        sampled_data = list(sampled_data)
        for i,d in enumerate(sampled_data):
            sampled_data[i]['audio']['raw'] = d['audio']['array']
        
        #Empty CUDA cache
        torch.cuda.empty_cache()

        #Wait 5 seconds
        time.sleep(5)
        #Initialize CarbonTracker
        tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
        tracker.epoch_start()
        for data in sampled_data:
            try:
                result = pipe(data['audio'])
            except:
                print(data['audio'])
                raise(0)

        #Capture energy
        timing, energy, divided = tracker.epoch_end()
        divided = [float(d) for d in divided]

        #Saving information
        energies.append({"tim": timing, "energy":energy, "divided":divided})
    info = {'num_params':num_params,'energies':energies}
    print(info)