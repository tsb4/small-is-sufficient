from diffusers import DiffusionPipeline
from  carbontracker.tracker import CarbonTracker
import json

#from diffusers import Dalle2Pipeline
import torch

model_name = "dreamlike-art/dreamlike-photoreal-2.0"
# model_name = "stabilityai/stable-diffusion-xl-base-1.0"
# model_name = "playgroundai/playground-v2.5-1024px-aesthetic"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
# Function to count parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Total parameter count for the entire pipeline
total_params = 0
print("Parameter count by component:")

# Dynamically go through components of the pipeline
for name, component in pipe.components.items():
    if hasattr(component, 'parameters'):  # Check if the component has parameters
        num_params = count_parameters(component)
        print(f"{name}: {num_params} parameters")
        total_params += num_params
print(f"Total parameters in DiffusionPipeline: {total_params/1e9}")
pipe = pipe.to("cuda")


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

energies = []
for _ in range(10):
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()
    image = pipe(prompt).images[0]
    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]
    energies.append({"tim": timing, "energy":energy, "divided":divided})

info = {'num_params':total_params,'energies':energies}


image.save("img.png")
