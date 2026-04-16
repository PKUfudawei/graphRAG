"""
RAG 系统主模块 - 使用 indexer 和 retriever 模块
"""

import os
import sys
import argparse
from typing import List
from glob import glob
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from rag.indexer import get_indexer, get_chunker, get_embedding
from rag.retriever import get_retriever
from models.llm import get_llm


def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG 系统")
    parser.add_argument("-b", "--build", type=str, help="构建索引：指定文件路径或 glob 模式")
    parser.add_argument("-q", "--query", type=str, help="单次查询")
    parser.add_argument("-i", "--interact", action="store_true", help="交互模式")
    parser.add_argument("-v", "--vectorstore", type=str, default="./data/vectorstore",
                        help="向量存储路径")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embed_device", type=str, default="cuda:0")
    parser.add_argument("--rerank_device", type=str, default="cuda:1")

    args = parser.parse_args()
    return args, parser


def build_index(files: List[str], indexer, vectorbase_path: str) -> None:
    """构建向量索引（支持纯文本文件）"""
    # 读取文件并创建 Document 对象
    documents = []
    for file_path in tqdm(files, desc="Loading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={"source": file_path}
        ))

    # 索引文档并构建向量库
    chunks = indexer.index_documents(documents)
    vectorstore = indexer.build_vectorstore(chunks)
    indexer.save_vectorstore(vectorstore, vectorbase_path)
    print(f"Built index with {len(chunks)} chunks, saved to {vectorbase_path}")


def generate_answer(query: str, context_docs: List[Document], llm=None) -> str:
    """根据上下文生成答案"""
    if llm is None:
        llm = get_llm()

    context_text = "\n\n".join(
        f"[Chunk {doc.metadata.get('chunk_id', 'unknown')}]: {doc.page_content}" for doc in context_docs
    )
    prompt = f"""根据以下参考信息回答问题。如果信息不足，请说明不知道。

参考信息:
{context_text}

问题：{query}

回答："""
    return llm.invoke([HumanMessage(content=prompt)]).content


def rag_query(query: str, retriever, llm=None, max_context_docs=None) -> dict:
    """RAG 查询"""
    docs = retriever.retrieve(query)
    context_docs = docs[:max_context_docs] if max_context_docs else docs
    answer = generate_answer(query, context_docs, llm)

    return {
        "query": query,
        "answer": answer,
        "references": [
            {"content": d.page_content[:200] + "..." if len(d.page_content) > 200 else d.page_content,
             "source": d.metadata.get("source", "unknown")}
            for d in context_docs
        ],
    }


def main():
    args, parser = parse_arguments()
    if not any([args.build, args.query, args.interact]):
        parser.print_help()
        return
    
    chunker = get_chunker(chunk_size=args.chunk_size, overlap=args.overlap, truncations=None)
    embedding = get_embedding(device=args.embed_device)
    indexer = get_indexer(chunker, embedding)

    # 构建模式
    if args.build:
        files = glob(args.build)
        if not files:
            print(f"Error: No files matched '{args.build}'")
            return
        build_index(files, indexer, args.vectorstore)
        return

    # 查询/交互模式
    if not os.path.exists(args.vectorstore):
        print(f"Error: Vectorstore not found at {args.vectorstore}")
        return

    vectorstore = indexer.load_vectorstore(args.vectorstore)
    n_chunks = len(vectorstore.docstore._dict.items())
    print(f"Loaded {n_chunks} chunks")
    retriever = get_retriever(
        vectorstore = vectorstore,
        top_k = 5,
        reranker_model = "BAAI/bge-reranker-v2-m3",
        reranker_device = args.rerank_device,
        rerank_top_k = None,
    )
    llm = get_llm(streaming=True)

    # 单次查询
    if args.query:
        result = rag_query(args.query, retriever, llm)
        print(f"Answer: {result['answer']}")
        return

    # 交互模式
    if args.interact:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                q = input("\nQuestion: ").strip()
                if q.lower() in ['quit', 'exit', 'q']:
                    break
                if not q:
                    continue
                result = rag_query(q, retriever, llm)
                print(f"Answer: {result['answer']}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
