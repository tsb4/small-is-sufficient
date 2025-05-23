from urllib.request import urlopen
from PIL import Image
import timm
import time
import json
import torch
from  carbontracker.tracker import CarbonTracker
import pandas as pd
import os



img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

#Test different models
models = [{'model':'timm/tiny_vit_21m_512.dist_in22k_ft_in1k'}, 
         {'model':"timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"}]

for model in models:
    model_name= model['model']
    model = timm.create_model(model_name, pretrained=True)
    model = model.eval()
    model = model.to("cuda")

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    num_params = sum(p.numel() for p in model.parameters())
    energies = []

    for _ in range(10):
        torch.cuda.empty_cache()
        time.sleep(5)
        tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
        tracker.epoch_start()
        output = model(transforms(img).unsqueeze(0).to("cuda"))  # unsqueeze single image into batch of 1
        timing, energy, divided = tracker.epoch_end()
        divided = [float(d) for d in divided]
        energies.append({"tim": timing, "energy":energy, "divided":divided})
    info = {'num_params':num_params,'energies':energies}
    print(info)
