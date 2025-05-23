from diffusers import DiffusionPipeline
from  carbontracker.tracker import CarbonTracker
import json
import torch

# Function to count parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Define model
model_name = "dreamlike-art/dreamlike-photoreal-2.0"

#Load Model
pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)


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

#Model to CUDA
pipe = pipe.to("cuda")

#Input prompt
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

energies = []

#testing multiple times
for _ in range(10):
    #Initialize CarbonTracker
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()

    #Inference
    image = pipe(prompt).images[0]

    #Capture energy
    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]

    #Saving information
    energies.append({"tim": timing, "energy":energy, "divided":divided})

info = {'num_params':total_params,'energies':energies}


image.save("img.png")
