# -*- coding: utf-8 -*-
import os
import json

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(current_dir)

with open(os.path.join(current_dir, 'dataset/val_dataset.json'), 'r', encoding="utf-8") as f:
    content = json.loads(f.read())

queries = list(content['queries'].values())
query_relevant_docs = {content['queries'][k]: v for k, v in content['relevant_docs'].items()}
node_id_text_mapping = content['corpus']
text_node_id_mapping = {v: k for k, v in node_id_text_mapping.items()}
