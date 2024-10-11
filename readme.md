
基于LLM的知识问答智能体的基于原理是先通过向量检索找到与用户query最相关的文档，然后将这些文档和任务指令拼接为一个prompt发送给LLM, 由LLM进行提炼总结最终的答案。
这种实现方式对于提升LLM在不熟悉的领域的问答方面具有显著优势，简单来说，它可以限定LLM的回答范围，避免幻觉式的回答，这种方法有助于LLM在一些比较专业的领域知识库进行问答。
实际应用中，利用预训练的文本嵌入模型(如bge/m3e/bce等模型)进行向量检索时，在专业知识领域检索效果未必很好，这是因为预训练的文本嵌入模型一般是在通用语料上进行训练，不能很好的表征专业知识领域文本特征。因而，为了提高向量检索的准确性，我们可以利用专业领域知识文档数据对预训练的文本嵌入模型进行微调，从而提升向量检索的准确性。

本项目基于llama_index框架，并参考了项目https://github.com/percent4/embedding_rerank_retrieval

需要注意的是embedding_rerank_retrieval使用llama_index框架在进行数据生成时调用了OpenAI的API实现，本项目对此进行了改进，使得llama_index可以完全使用本地模型进行全流程的开发，从而可以降低开发成本并提升了数据安全。

一、训练阶段：
(1)数据预处理：对专业语料数据进行切分为后续数据生成提供材料，并划分为训练集和验证集。
(2)数据生成：使用预处理的数据块基于LLM生成问答对。
(3)训练嵌入模型：使用生成的问答对数据对文本嵌入模型进行训练。

二、切分方式：
为了让LLM更好的汇总答案，我们尽可能的采用具有完整语义的段落划分的方式。假如原word文档具有标题, 我们可以把标题和相邻标题之间的内容作为一个整体段落。
例如：
标题1
[内容1]
标题1.1
[内容1.1]
标题1.2
[内容1.2]

切分后得到如下文档块：
标题1：[内容1]
标题1.1：[内容1.1]
标题1.2：[内容1.2]

切分时有的文档块可能内容过长，这时我们才会将过长的文档可切分为多个子块，并且相邻块之间有一定的字数重叠，帮助LLM发现它们属于同一个文档块。

三、项目使用方法(步骤)

pip3 install -r requirements.txt
(实际使用时不一定要与这些软件版本完全一致，能跑起来就行)

cd embedding_rag

(1) 切分原文档得到数据集

python3 doc/wordconvert2txt.py

(这个例子中的原始文档由于格式问题没有heading style，对整篇文档进行了等长度的切分。)

(2)生成训练数据

python3 dataset/dataset_pro.py

(3)训练文本嵌入模型
设置embedding_finetune/train.py中的训练集、验证集、预训练嵌入模型、微调后的嵌入模型的路径：

python3 embedding_finetune/train.py

(4)评估前准备
部署模型：

python3 embedding_service/embedding_server.py

验证集的嵌入向量：

python3 evaluate/build_embedding_cache.py

(5)评估嵌入模型
与embedding_rerank_retrieval里的评估方法一样，这里通过计算召回命中率等指标只对嵌入模型进行评估。

python3 evaluate/evaluation.py




English version 
### 
The principle behind the intelligent agent for knowledge-based question answering based on Large Language Models (LLMs) involves first using vector retrieval to find the most relevant documents to the user's query. These documents are then concatenated with task instructions into a prompt, which is sent to the LLM for processing. The LLM refines and summarizes the information to produce the final answer.

This approach has significant advantages in improving the LLM's ability to answer questions in unfamiliar domains. Simply put, it restricts the LLM's response scope, avoiding hallucinatory answers. This method helps the LLM perform question-and-answer tasks within more specialized knowledge bases.

In practical applications, when using pre-trained text embedding models (such as bge/m3e/bce models) for vector retrieval, the search results in professional fields may not be optimal. This is because these pre-trained text embedding models are generally trained on general corpora and cannot well represent the textual features of specialized knowledge areas. Therefore, to improve the accuracy of vector retrieval, we can fine-tune the pre-trained text embedding models using data from professional knowledge documents, thereby enhancing the accuracy of vector retrieval.

This project is based on the llama_index framework and references the project https://github.com/percent4/embedding_rerank_retrieval.

It is worth noting that embedding_rerank_retrieval uses the llama_index framework to generate data by calling OpenAI's API. This project improves upon this by enabling the entire development process to use local models with the llama_index framework, thus reducing development costs and enhancing data security.

### Training Phase:
(1) Data Preprocessing
Segment professional corpus data to provide material for subsequent data generation, and divide it into training and validation sets.
(2) Data Generation
Generate QA pairs based on preprocessed data chunks using an LLM.
(3) Train Embedding Model
Use the generated QA pair data to train the text embedding model.

### Segmentation Method:
To better enable the LLM to summarize answers, we adopt a segmentation method that preserves complete semantic units wherever possible. If the original Word document contains headings, we can treat the content between adjacent headings as a single segment.
For example:
Heading 1
[Content 1]
Subheading 1.1
[Content 1.1]
Subheading 1.2
[Content 1.2]

After segmentation, the resulting document chunks would be:
Heading 1: [Content 1]
Subheading 1.1: [Content 1.1]
Subheading 1.2: [Content 1.2]

If some document chunks are too long, they can be further divided into sub-chunks, with overlapping text between adjacent segments to help the LLM recognize they belong to the same document chunk.

### Project Usage Steps

pip3 install -r requirements.txt

(Note: In actual use, the software versions do not necessarily need to match exactly; they just need to work.)

cd embedding_rag

(1) Segment the original document to obtain the dataset

python3 doc/wordconvert2txt.py

(In this example, due to formatting issues, the original document did not have heading styles, so the entire document was segmented into equal lengths.)

(2) Generate training data

python3 dataset/dataset_pro.py

(3) Train Text Embedding Model
Set the paths for the training set, validation set, pre-trained embedding model, and the path to the fine-tuned embedding model in embedding_finetune/train.py:

python3 embedding_finetune/train.py

(4) Preparation Before Evaluation
Deploy the model: python3 embedding_service/embedding_server.py
Generate embeddings for the validation set:

python3 evaluate/build_embedding_cache.py

(5) Evaluate the Embedding Model
Evaluation method is the same as project embedding_rerank_retrieval, here only evaluate the embedding model.

python3 evaluate/evaluation.py

