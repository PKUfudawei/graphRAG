import json
import os
from typing import List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class Chunker:
    """使用 LangChain 框架的文本分块器"""

    def __init__(self, chunk_size: int, overlap: int, embed_model: Optional[str] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        self.embeddings = None
        if embed_model is not None:
            self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    def chunk_text(self, text: str, source: str = None) -> List[Document]:
        """将文本分割成 chunks"""
        documents = self.text_splitter.create_documents([text], metadatas=[{"source": source}])
        return documents

    def embed_chunks(self, documents: List[Document]) -> List[Document]:
        """为 chunks 生成 embedding"""
        if self.embeddings is None:
            print("No embedding model specified, skipping embedding step")
            return documents
        # LangChain 的 embeddings 会在 vectorstore 中自动调用
        return documents

    def save_chunks(self, documents: List[Document], path: str):
        """保存 chunks 到文件"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        chunks_data = [
            {
                'id': idx,
                'text': doc.page_content,
                'source': doc.metadata.get('source'),
                'metadata': doc.metadata,
            }
            for idx, doc in enumerate(documents)
        ]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"\tSaved {len(chunks_data)} chunks to {path}")

    def load_chunks(self, path: str) -> List[Document]:
        """从文件加载 chunks"""
        with open(path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        documents = [
            Document(page_content=data['text'], metadata=data.get('metadata', {}))
            for data in chunks_data
        ]
        print(f"Loaded {len(documents)} chunks from {path}")
        return documents

    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """使用 LangChain 构建 FAISS vectorstore"""
        if self.embeddings is None:
            raise ValueError("Embedding model must be specified to build vectorstore")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore

    def save_vectorstore(self, vectorstore: FAISS, path: str):
        """保存 vectorstore"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        vectorstore.save_local(path)
        print(f"\tVectorstore saved to {path}")

    def load_vectorstore(self, path: str) -> FAISS:
        """加载 vectorstore"""
        vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"\tVectorstore loaded from {path}")
        return vectorstore

    def chunk_file(self, path: str, chunks_path: str = None,
                   vectorstore_path: str = None) -> List[Document]:
        """处理单个文件"""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        documents = self.chunk_text(text, source=path)

        if chunks_path is not None:
            self.save_chunks(documents, chunks_path)
        if vectorstore_path is not None:
            vectorstore = self.build_vectorstore(documents)
            self.save_vectorstore(vectorstore, vectorstore_path)

        return documents

    def chunk_files(self, files: List[str], chunks_path: str = None,
                    vectorstore_path: str = None, max_workers: int = 16) -> List[Document]:
        """并行处理多个文件"""
        all_documents = []

        def process_file(path: str) -> List[Document]:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.chunk_text(text, source=path)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in files}
            for future in tqdm(as_completed(futures), total=len(files), desc="Chunking files"):
                file = futures[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

        if chunks_path is not None:
            self.save_chunks(all_documents, chunks_path)

        if vectorstore_path is not None:
            vectorstore = self.build_vectorstore(all_documents)
            self.save_vectorstore(vectorstore, vectorstore_path)

        return all_documents


def test_chunker():
    """测试 Chunker 类的功能"""
    import tempfile
    import shutil

    # 测试文本
    test_text = """
    这是一段测试文本，用于测试文本分块器的功能。
    这是第二段内容，包含更多的信息。
    这是第三段内容，继续添加更多细节。
    这是第四段内容，测试分块器是否能正确分割文本。
    这是第五段内容，确保分块器工作正常。
    """ * 10  # 重复多次以生成长文本

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 测试 1: 基本分块功能
        print("=" * 50)
        print("测试 1: 基本分块功能")
        print("=" * 50)
        chunker = Chunker(chunk_size=200, overlap=50)
        documents = chunker.chunk_text(test_text, source="test_source")
        print(f"生成了 {len(documents)} 个 chunks")
        for i, doc in enumerate(documents[:3]):
            print(f"\nChunk {i}: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")

        # 测试 2: 保存和加载 chunks
        print("\n" + "=" * 50)
        print("测试 2: 保存和加载 chunks")
        print("=" * 50)
        chunks_path = os.path.join(temp_dir, "chunks.json")
        chunker.save_chunks(documents, chunks_path)
        loaded_docs = chunker.load_chunks(chunks_path)
        print(f"加载了 {len(loaded_docs)} 个 chunks")
        assert len(documents) == len(loaded_docs), "保存和加载的 chunks 数量不一致"

        # 测试 3: 文件分块
        print("\n" + "=" * 50)
        print("测试 3: 文件分块")
        print("=" * 50)
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_text)
        file_docs = chunker.chunk_file(test_file, chunks_path=os.path.join(temp_dir, "file_chunks.json"))
        print(f"文件分块生成了 {len(file_docs)} 个 chunks")

        # 测试 4: 多文件并行处理
        print("\n" + "=" * 50)
        print("测试 4: 多文件并行处理")
        print("=" * 50)
        test_files = []
        for i in range(3):
            tf = os.path.join(temp_dir, f"test_{i}.txt")
            with open(tf, 'w', encoding='utf-8') as f:
                f.write(test_text * (i + 1))
            test_files.append(tf)
        multi_docs = chunker.chunk_files(test_files, chunks_path=os.path.join(temp_dir, "multi_chunks.json"))
        print(f"多文件处理生成了 {len(multi_docs)} 个 chunks")

        # 测试 5: 不同 chunk_size 的效果
        print("\n" + "=" * 50)
        print("测试 5: 不同 chunk_size 的效果")
        print("=" * 50)
        for size in [100, 200, 500]:
            chunker_small = Chunker(chunk_size=size, overlap=20)
            docs = chunker_small.chunk_text(test_text, source="test")
            print(f"chunk_size={size}: 生成 {len(docs)} 个 chunks")

        print("\n" + "=" * 50)
        print("所有测试通过!")
        print("=" * 50)

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"清理临时目录：{temp_dir}")


if __name__ == "__main__":
    test_chunker()