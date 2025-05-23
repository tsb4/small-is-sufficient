
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

#Define models and dataset
models = ["jozhang97/deta-resnet-50-24-epochs"]
load_coco_dataset()

for idx_model,model_name in enumerate(models):
    # Load dataset
    dataset = load_test_dataset_object_detection()
    #Load Model
    object_detection_pipeline = pipeline("object-detection", model=model_name, device='cuda')
    num_params = object_detection_pipeline.model.num_parameters()
    energies = []
    
    #Testing multiple times
    for _ in range(10):
        #Empty CUDA cache
        torch.cuda.empty_cache()
        #Wait 5 seconds
        time.sleep(5)
        st = time.time()
        #Initialize CarbonTracker
        tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
        tracker.epoch_start()
        for i in range(1):
            result = object_detection_pipeline(dataset)
        ed = time.time()
        #capture energy
        timing, energy, divided = tracker.epoch_end()
        divided = [float(d) for d in divided]
        #Saving information
        energies.append({"tim": timing, "energy":energy, "divided":divided, "timestamp_start":st, "timestamp_end":ed})
    info = {'num_params':num_params,'energies':energies}
    print(info)