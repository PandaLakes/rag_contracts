from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# loading models and data
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device=device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# loading the embeddings and "pages and chunks" data
data_embeddings = pd.read_csv("text_chunks_embeddings_df.csv")
pages_and_chunks = data_embeddings.to_dict(orient="records")
data_embeddings['embedding'] = data_embeddings['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
embeddings = torch.tensor(np.stack(data_embeddings['embedding'].tolist(), axis=0), dtype=torch.float32).to(device)

# initialize FastAPI
app = FastAPI()

class QueryModel(BaseModel):
    query: str


def retrieve(query: str, embeddings: torch.tensor, model: SentenceTransformer, num_resources_to_return: int = 2):
    query_embedding = model.encode(query, convert_to_tensor=True)
    dot_score = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_score, k=num_resources_to_return)
    return scores, indices

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentences_chunks"] for item in context_items])
    base_prompt = f"""Based on the following context items, generate a comprehensive and well-structured contract. Ensure the contract covers all necessary elements and addresses the query effectively.
Context items:
{context}
Query:{query}
Answer:
"""
    dialogue_template = [{"role": "user", "content": base_prompt}]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
    return prompt

@app.post("/ask/")
async def ask_rag(query: QueryModel):
    try:
        query_text = query.query
        scores, indices = retrieve(query=query_text, embeddings=embeddings, model=model)
        context_items = [pages_and_chunks[i] for i in indices]
        prompt = prompt_formatter(query=query_text, context_items=context_items)
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = llm_model.generate(**input_ids, temperature=0.7, do_sample=True, max_new_tokens=256)
        output_text = tokenizer.decode(outputs[0]).replace(prompt, "").replace("<bos>", "").replace("<eos>", "")
        return {"query": query_text, "answer": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
