from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from  carbontracker.tracker import CarbonTracker
import json
import bitsandbytes as bnb  # Ensure bitsandbytes is installed for FP8 quantization
import os
import time

quantization_config = BitsAndBytesConfig(load_in_8bit=True)




#Define Model
models = ["microsoft/Phi-3-mini-128k-instruct"]  # Replace with the exact model name




for model_name in models:
    print("Model name: ", model_name)

    #Load Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # FP8 is generally emulated using FP16
        device_map="cuda",
        quantization_config=quantization_config  # 8-bit quantization
    )
    

    num_parameters = sum(p.numel() for p in model.parameters())


    # Define the input prompt
    prompt = "Write a 50-token story about a futuristic world where AI and humans coexist:"

    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    energies = []

    #Testing Multiple times
    for i in range(10):
        # Wait 5 seconds
        time.sleep(5)
        #Empty CUDA cache
        torch.cuda.empty_cache()
        
        #Initialize CarbonTracker
        tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
        tracker.epoch_start()
        
        # Generate text with exact token limit
        max_tokens = 50  # Generate exactly 50 tokens
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,  # To allow variability in generation
            pad_token_id=tokenizer.eos_token_id  # Avoid padding issues
        )

        # Decode the generated tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract and print the last 100 tokens
        generated_tokens = tokenizer.tokenize(generated_text)
        
        # Capture energy
        timing, energy, divided = tracker.epoch_end()
        divided = [float(d) for d in divided]
        
        # Saving information
        energies.append({"tim": timing, "energy":energy, "divided":divided, "token": len(generated_tokens)})
    info = {'num_params':num_parameters,'energies':energies}
