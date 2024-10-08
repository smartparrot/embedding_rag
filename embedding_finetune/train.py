# coding = utf-8
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.finetuning import EmbeddingQAFinetuneDataset

train_dataset = EmbeddingQAFinetuneDataset.from_json("dataset/train_dataset.json") # 训练集
val_dataset = EmbeddingQAFinetuneDataset.from_json("dataset/val_dataset.json") # 验证集

#--------start training---------
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="/workspace/host_data/models/m3e-base", # 改为预训练文本嵌入模型路径
    model_output_path="/workspace/host_data/models/m3e-ft-001", # 微调后的模型存储路径
    val_dataset=val_dataset,
)

print(finetune_engine.loss)
finetune_engine.finetune()
