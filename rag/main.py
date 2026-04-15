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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer import get_indexer
from models.reranker import get_reranker
from models.llm import get_llm


def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG 系统")
    parser.add_argument("-b", "--build", type=str, help="构建索引：指定文件路径或 glob 模式")
    parser.add_argument("-q", "--query", type=str, help="单次查询")
    parser.add_argument("-i", "--interact", action="store_true", help="交互模式")
    parser.add_argument("-v", "--vectorstore", type=str, default="../data/vectorstore",
                        help="向量存储路径")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embed_device", type=str, default="cuda:0")
    parser.add_argument("--rerank_device", type=str, default="cuda:1")

    args = parser.parse_args()
    return args, parser


def get_retriever(vectorstore, top_k=5, reranker_device="cuda:0"):
    """创建检索器"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    reranker = get_reranker(device=reranker_device, top_k=top_k)

    def retrieve(query: str):
        docs = retriever.invoke(query)
        if docs:
            docs = reranker.rerank(query, docs)
        return docs

    retrieve.retrieve = retrieve
    return retrieve


def build_index(files: List[str], vectorbase_path: str, embed_device: str = "cuda:0",
                chunk_size: int = 512, overlap: int = 50) -> None:
    """构建向量索引（支持纯文本文件）"""
    indexer = get_indexer(chunk_size=chunk_size, overlap=overlap, embed_device=embed_device)

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


def main():
    args, parser = parse_arguments()
    if not any([args.build, args.query, args.interact]):
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
