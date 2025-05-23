from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import AutoImageProcessor, AutoModel

from PIL import Image
import requests
from  carbontracker.tracker import CarbonTracker
import json, torch, time
#Load Model
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

#Model to CUDA
model = model.to("cuda")

#Open image to test
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").to("cuda")

num_params = sum(p.numel() for p in model.parameters())
energies = []

for _ in range(10):
    #Empty CUDA cache
    torch.cuda.empty_cache()
    #Wait 5 seconds
    time.sleep(5)
    #Initialize CarbonTracker
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()

    outputs = model(**inputs)
    
    #Capture Energy
    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]
    #Saving Information
    energies.append({"tim": timing, "energy":energy, "divided":divided})

info = {'num_params':num_params,'energies':energies}
print(info)

