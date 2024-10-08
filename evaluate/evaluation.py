# -*- coding: utf-8 -*-
import asyncio
import time

import pandas as pd
from datetime import datetime
from faiss import IndexFlatIP
from llama_index.evaluation import RetrieverEvaluator
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
from vector_store_retriever import VectorSearchRetriever

# Display results from evaluate.
def display_results(name_list, eval_results_list):
    pd.set_option('display.precision', 4)
    columns = {"retrievers": [], "hit_rate": [], "mrr": []}
    for name, eval_results in zip(name_list, eval_results_list):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        columns["retrievers"].append(name)
        columns["hit_rate"].append(hit_rate)
        columns["mrr"].append(mrr)

    metric_df = pd.DataFrame(columns)
    return metric_df

doc_qa_dataset = EmbeddingQAFinetuneDataset.from_json("dataset/val_dataset.json")
metrics = ["mrr", "hit_rate"]

# embedding retrieve
evaluation_name_list = []
evaluation_result_list = []
cost_time_list = []

for top_k in [1, 2, 3, 4, 5]:
    start_time = time.time()
    faiss_index = IndexFlatIP(768)
    embedding_retriever = VectorSearchRetriever(top_k=top_k, faiss_index=faiss_index)
    embedding_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=embedding_retriever)
    embedding_eval_results = asyncio.run(embedding_retriever_evaluator.aevaluate_dataset(doc_qa_dataset))
    evaluation_name_list.append(f"embedding_top_{top_k}_eval")
    evaluation_result_list.append(embedding_eval_results)
    faiss_index.reset()
    cost_time_list.append((time.time() - start_time) * 1000)

print("done for embedding evaluation!")
df = display_results(evaluation_name_list, evaluation_result_list)
df['cost_time'] = cost_time_list
print(df.head())
df.to_csv(f"evaluation_m3e_embedding_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)