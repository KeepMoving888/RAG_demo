"""
文档处理模块：支持PDF、Word、TXT等多种格式
"""
import os
from typing import List, Dict
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
from docx import Document as WordDocument


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
    
    def read_pdf(self, file_path: str) -> str:
        """读取PDF文件"""
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def read_word(self, file_path: str) -> str:
        """读取Word文件"""
        doc = WordDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    def read_txt(self, file_path: str) -> str:
        """读取TXT文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_document(self, file_path: str) -> str:
        """根据文件类型加载文档"""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return self.read_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.read_word(file_path)
        elif ext == '.txt':
            return self.read_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")
    
    def process_file(self, file_path: str, metadata: Dict = None) -> List[Document]:
        """处理单个文件，返回分块后的文档列表"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        text = self.load_document(file_path)
        
        if metadata is None:
            metadata = {
                "source": os.path.basename(file_path),
                "file_path": file_path
            }
        
        docs = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata] * len([text])
        )
        
        # 为每个chunk添加位置信息
        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = f"{metadata['source']}_{i}"
            doc.metadata["chunk_index"] = i
        
        return docs
    
    def process_directory(self, dir_path: str) -> List[Document]:
        """处理目录下所有文档"""
        all_docs = []
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                try:
                    docs = self.process_file(file_path)
                    all_docs.extend(docs)
                    print(f"✅ 处理完成：{file_name} ({len(docs)} chunks)")
                except Exception as e:
                    print(f"❌ 处理失败：{file_name}, 错误：{e}")
        return all_docs


if __name__ == "__main__":
    # 测试文档处理
    processor = DocumentProcessor()
    docs = processor.process_directory("./data/documents")
    print(f"总共处理了 {len(docs)} 个文档块")