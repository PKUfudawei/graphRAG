"""
RAG 系统主模块 - 使用 index/ 和 retrieve/ 模块
"""

import os
import sys
import argparse
from typing import List
from glob import glob
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from index import get_indexer
from retrieve import get_retriever
from llm import get_llm


def build_index(files: List[str], vectorbase_path: str, embed_device: str = "cuda:0",
                chunk_size: int = 512, overlap: int = 50) -> None:
    """构建向量索引（支持 Markdown 和纯文本）"""
    indexer = get_indexer(chunk_size=chunk_size, overlap=overlap, embed_device=embed_device)

    # 使用 index_files 方法，自动支持 Markdown 和纯文本
    documents, vectorstore = indexer.index_files(files, build_vectorstore=True)
    indexer.save_vectorstore(vectorstore, vectorbase_path)
    print(f"Built index with {len(documents)} chunks, saved to {vectorbase_path}")


def generate_answer(query: str, context_docs: List[Document], llm=None) -> str:
    """根据上下文生成答案"""
    if llm is None:
        llm = get_llm()

    context_text = "\n\n".join(
        f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(context_docs)
    )
    prompt = f"""根据以下参考信息回答问题。如果信息不足，请说明不知道。

参考信息:
{context_text}

问题：{query}

回答："""
    return llm.invoke([HumanMessage(content=prompt)]).content


def rag_query(query: str, retriever, llm=None, max_context_docs: int = 3) -> dict:
    """RAG 查询"""
    docs = retriever.retrieve(query)
    context_docs = docs[:max_context_docs]
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG 系统")
    parser.add_argument("--build", type=str, help="构建索引：指定文件路径或 glob 模式")
    parser.add_argument("--query", type=str, help="单次查询")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--vectorstore", type=str, default="./data/vectorstore",
                        help="向量存储路径")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-device", type=str, default="cuda:0")
    parser.add_argument("--rerank-device", type=str, default="cuda:1")

    args = parser.parse_args()
    return args, parser


def main():
    args, parser = parse_arguments()
    if not any([args.build, args.query, args.interactive]):
        parser.print_help()
        return

    # 构建模式
    if args.build:
        files = glob(args.build)
        if not files:
            print(f"Error: No files matched '{args.build}'")
            return
        build_index(files, args.vectorstore, args.embed_device, args.chunk_size, args.overlap)
        return

    # 查询/交互模式
    if not os.path.exists(args.vectorstore):
        print(f"Error: Vectorstore not found at {args.vectorstore}")
        return

    indexer = get_indexer(embed_device=args.embed_device)
    vectorstore = indexer.load_vectorstore(args.vectorstore)
    retriever = get_retriever(vectorstore, top_k=args.top_k, reranker_device=args.rerank_device)
    llm = get_llm(model="Qwen/Qwen3.5-27B", base_url="http://localhost:8000/v1", api_key="EMPTY")

    # 单次查询
    if args.query:
        result = rag_query(args.query, retriever, llm)
        print(f"Answer: {result['answer']}")
        return

    # 交互模式
    if args.interactive:
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
