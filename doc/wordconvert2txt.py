# coding: utf-8
from docx import Document

def is_heading(paragraph, heading_style_names):
    # 检查当前段落是否是标题
    return paragraph.style.name in heading_style_names

def concatenate_paragraphs(paragraphs):
    # 连接多个段落为一个字符串，并保留换行�?
    return '\n'.join([p.text for p in paragraphs])

def split_paragraph(paragraph, title, max_length=512, overlap_length=50):
    if overlap_length >= max_length:
        raise ValueError("Overlap length must be less than max length")
    
    chunks = []
    start = 0
    while start < len(paragraph):
        end = min(start + max_length, len(paragraph))
        
        chunks.append((title, paragraph[start:end]))

        if end < len(paragraph):
            start = end - overlap_length
        else:
            break
    
    return chunks


def read_word_document(file_name, file_path, heading_style_names=None):
    if heading_style_names is None:
        heading_style_names = ['Heading 1', 'Heading 2', 'Heading 3']  

    doc = Document(file_path)
    result = []
    current_title = ""
    current_paragraphs = []
    
    for paragraph in doc.paragraphs:
        
        if is_heading(paragraph, heading_style_names): # encounter the seond title, combine the first title and contents
            if current_paragraphs:
                # 当前段落已收集完毕，加入结果列表
                result.append((file_name, current_title, concatenate_paragraphs(current_paragraphs)))
                current_paragraphs = []  # 清空当前段落
            current_title = paragraph.text.strip()
            current_paragraphs = []  # 初始化当前段落列�?
        else: # encounter the first title
            # 如果不是标题，则将非空段落添加到当前段落列表�?
            if paragraph.text.strip():
                current_paragraphs.append(paragraph)
    
    # 处理最后一个段�?
    if current_paragraphs:
        result.append((file_name, current_title, concatenate_paragraphs(current_paragraphs)))

    return result


def generate_txt_files(results, max_length=512, overlap_length=50):
    for file_name, title, paragraph in results:
        if len(paragraph) > max_length:
            sub_paragraphs = split_paragraph(paragraph, title, max_length=max_length, overlap_length=overlap_length)
            for j, (sub_title, sub_paragraph) in enumerate(sub_paragraphs):
                with open(f"doc/dataset/{file_name}_{j+1}.txt", "w", encoding="utf-8") as f:
                    f.write(f"# 文件索引: {file_name}\n")
                    f.write(f"# 段落标题: {sub_title}\n")
                    f.write(f"# 原始段落位置: {sub_title}（第{j+1}部分）\n")
                    f.write(f"# 是否完整段落: 否\n")
                    f.write(f"# 段落内容:\n{sub_paragraph}\n")
        else:
            with open(f"doc/dataset/{file_name}.txt", "w", encoding="utf-8") as f:
                f.write(f"# 文件索引: {file_name}\n")
                f.write(f"# 段落标题: {title}\n")
                f.write(f"# 原始段落位置: {title}\n")
                f.write(f"# 是否完整段落: 是\n")
                f.write(f"# 段落内容:\n{paragraph}\n")

if __name__ == "__main__":
    input_file = "doc/xiyouji.docx"
    file_name = "xiyouji.docx"
    heading_styles = ['Heading 1', 'Heading 2', 'Heading 3'] 
    paragraphs = read_word_document(file_name, input_file, heading_styles)
    generate_txt_files(paragraphs, max_length=512, overlap_length=50)