from transformers import AutoFeatureExtractor, ASTForAudioClassification
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from  carbontracker.tracker import CarbonTracker
import json, torch, time
from datasets import load_dataset
import torch

# Model example

#Load Model
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = ASTForAudioClassification.from_pretrained(model_name)

#Model to CUDA
model = model.to("cuda")

num_params = sum(p.numel() for p in model.parameters())
energies = []

#Test 10 times
for _ in range(10):
    #Load dataset
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
    dataset = dataset.sort("id")

    sampling_rate = dataset.features["audio"].sampling_rate

    #Empty CUDA memory
    torch.cuda.empty_cache()
    #Wait 5 secs
    time.sleep(5)
    #Initialize CarbonTracker
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()

    for i in range(10):
        inputs = extractor(dataset[i]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt").to("cuda")
        try:
            with torch.no_grad():
                logits = model(**inputs).logits
        except:
            raise(0)
    #Capture Energy
    
    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]
    energies.append({"tim": timing, "energy":energy, "divided":divided})
info = {'num_params':num_params,'energies':energies}
print(info)