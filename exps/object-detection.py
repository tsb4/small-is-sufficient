
import json
import os
import subprocess
from transformers import pipeline
from datasets import load_dataset

import torch
from PIL import Image
import requests
import argparse
import time
import csv
from  carbontracker.tracker import CarbonTracker

from PIL import Image
import io
from datasets import load_dataset


IDLE = False

def load_coco_dataset():
    if not os.path.exists("datasetval"):
        os.makedirs("datasetval")
    # Load the COCO2017 dataset
    dataset = load_dataset("rafaelpadilla/coco2017", split="val", streaming=True)
    iterator = iter(dataset)
    sample = next(iterator)  # Get the first image in the test split
    ds = []
    for i in range(100):
        ds.append(sample['image'])
        sample = next(iterator)
    # Assuming `ds` is a list of images in bytes format or in some format convertible to PIL Image
    for idx, image in enumerate(ds):
        # Convert image data to PIL Image
        #image = Image.open(io.BytesIO(image_data))
        
        # Save the image to a file
        image.save(f"datasetval/image_{idx}.png")  # or .jpg or any other format


def load_test_dataset_object_detection():
    # Load the COCO2017 dataset
    ds = []
    for filename in sorted(os.listdir("datasetval/")):
        if filename.endswith('.png'):  # or .jpg or any other format
            image_path = os.path.join("datasetval/", filename)
            with Image.open(image_path) as img:
                ds.append(img.copy())
    return ds 

# with open("dict_leaderboards/objectDetection.json", "r") as file:
#     data_models = json.load(file)
# data_models = sorted(data_models, key=lambda x:x['nParams'], reverse=False)
# data_models = [d for d in data_models if d['nParams']>0 and d['nParams']<5e9]
models = ["jozhang97/deta-resnet-50-24-epochs"]
load_coco_dataset()

# Iterate over model names and print the number of parameters
for idx_model,model_name in enumerate(models):
    dataset = load_test_dataset_object_detection()
    object_detection_pipeline = pipeline("object-detection", model=model_name, device='cuda')
    num_params = object_detection_pipeline.model.num_parameters()
    energies = []
    

    for _ in range(10):
        torch.cuda.empty_cache()
        time.sleep(5)
        st = time.time()
        tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
        tracker.epoch_start()
        for i in range(1):
            result = object_detection_pipeline(dataset)
        ed = time.time()
        timing, energy, divided = tracker.epoch_end()
        divided = [float(d) for d in divided]
        energies.append({"tim": timing, "energy":energy, "divided":divided, "timestamp_start":st, "timestamp_end":ed})
    info = {'num_params':num_params,'energies':energies}
    print(info)