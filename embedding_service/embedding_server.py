# -*- coding: utf-8 -*-
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = FastAPI()
# model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
# model = SentenceTransformer('.../ft_models/bce-embedding-base_v1-ft-001')
model = SentenceTransformer('/workspace/host_data/models/m3e-base') 

class Sentence(BaseModel):
    text: str

@app.get('/')
def home():
    return 'hello world'

@app.post('/embedding')
def get_embedding(sentence: Sentence):
    embedding = model.encode(sentence.text, normalize_embeddings=True).tolist()
    return {"text": sentence.text, "embedding": embedding}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=50072)
