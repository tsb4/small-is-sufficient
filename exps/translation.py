from transformers import T5Tokenizer, T5ForConditionalGeneration
from  carbontracker.tracker import CarbonTracker
import json, torch, time

# Load the tokenizer and model

model_name = 'google-t5/t5-11b'
model_name = 'google-t5/t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name.split('/')[1])
model = T5ForConditionalGeneration.from_pretrained(model_name.split('/')[1])
model=model.to("cuda")


# Prepare the input text
input_text = "translate English to German: I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period."# Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful. You have requested a debate on this subject in the course of the next few days, during this part-session. In the meantime, I should like to observe a minute' s silence, as a number of Members have requested, on behalf of all the victims concerned, particularly those of the terrible storms, in the various countries of the European Union."
# input_text = "translate English to German: This is a beautiful word."
input_ids = tokenizer.encode(input_text, return_tensors='pt').to("cuda")

num_params = sum(p.numel() for p in model.parameters())
energies = []
for _ in range(10):
    torch.cuda.empty_cache()
    time.sleep(5)
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()
    #print(data)
    try:
        outputs = model.generate(input_ids, max_length=40, num_beams=4, early_stopping=True)

    except:
        raise(0)

    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]
    energies.append({"tim": timing, "energy":energy, "divided":divided})
info = {'num_params':num_params,'energies':energies}
print(info)

# Generate the output
