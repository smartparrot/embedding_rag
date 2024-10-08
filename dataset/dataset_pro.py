import json
import random
random.seed(42)
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import MetadataMode
from llama_index.schema import TextNode
from llama_index.llms.vllm import Vllm
from llama_index.finetuning import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset,
)
import os

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()

    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:

        print(f"Parsed {len(nodes)} nodes")
    return nodes


def create_text_nodes_from_txt_files(directory_path):
    text_nodes = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
                
                node_id = f"{filename}_id" 
                text_node = TextNode(text=text_content, id_=node_id)
                
                text_nodes.append(text_node)

    return text_nodes    


def split_data(nodes, k):
    if not (0 < k < 1):
        raise ValueError("k must be between 0 and 1")

    total_nodes = len(nodes)

    val_size = int(total_nodes * k)

    indices = list(range(total_nodes))
    
    random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    val_nodes = [nodes[i] for i in val_indices]
    train_nodes = [nodes[i] for i in train_indices]

    return train_nodes, val_nodes


def main():
    #-------prepare data-----------
    dataset = "doc/dataset"
    nodes = create_text_nodes_from_txt_files(dataset)

    k = 0.3 

    train_nodes, val_nodes = split_data(nodes, k)

    print(f"Train nodes count: {len(train_nodes)}")
    print(f"Validation nodes count: {len(val_nodes)}")

    print(train_nodes[:3])

    #-------process data-----------
    llm = Vllm(
        model="/workspace/host_data/models/qwen/Qwen2-7B-Instruct",
        tensor_parallel_size=1,
        max_new_tokens=1024,
        vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.7},
    )

    qa_generate_prompt_tmpl = """\
    <|im_start|>user
    Context information is below.
    ---------------------
    {context_str}
    --------------------
    Given the context information and not prior knowledge.
    generate only questions based on the below query.
    You are a Professor. Your task is to setup \
    {num_questions_per_chunk} questions for an upcoming \
    quiz/examination in Chinese. The questions should be diverse in nature \
    across the document in Chinese. The questions should not contain options, not start with Q1/Q2/number. \
    The questions should be meanful sentence.\
    Restrict the questions to the context information provided. 
    Abbreviations may be used for titles and professional terms.
    <im_end>
    <|im_start|>assistant\n
    """
    

    train_dataset = generate_qa_embedding_pairs(nodes=train_nodes, llm=llm, num_questions_per_chunk=3, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
    val_dataset = generate_qa_embedding_pairs(nodes=val_nodes, llm=llm, num_questions_per_chunk=3, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
    
    train_dataset.save_json("dataset/train_dataset.json")
    val_dataset.save_json("dataset/val_dataset.json")


main()











































