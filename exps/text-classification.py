from sentence_transformers import SentenceTransformer
from  carbontracker.tracker import CarbonTracker
import torch
import json

query_prompt_name = "s2p_query"

instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
prompt = f'<instruct>{instruction}\n<query>'

queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
# docs do not need any prompts
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]


model_name = "dunzhang/stella_en_400M_v5"
# model_name = "BAAI/bge-multilingual-gemma2"
model = SentenceTransformer(model_name, trust_remote_code=True,model_kwargs={"torch_dtype": torch.float16})
num_params = sum(p.numel() for p in model.parameters())
print("Num Parameters: ", num_params)
model = model.to("cuda")


energies = []
for _ in range(5):
    print("Start")
    tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
    tracker.epoch_start()
    for i in range(100):
        query_embeddings = model.encode(queries, prompt=prompt)#prompt_name=query_prompt_name)
        doc_embeddings = model.encode(docs)
        similarities = model.similarity(query_embeddings, doc_embeddings)
    timing, energy, divided = tracker.epoch_end()
    divided = [float(d) for d in divided]
    energies.append({"tim": timing, "energy":energy, "divided":divided})
info = {'num_params':num_params,'energies':energies}
print(info)
