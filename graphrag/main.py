"""
GraphRAG 系统主模块 - 使用 GraphRAGIndexer 和 GraphRAGRetriever
"""

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Add paths
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from graphrag.indexer import get_graphrag_indexer
from graphrag.retriever import get_graphrag_retriever
from models.llm import get_llm


def parse_arguments():
    parser = argparse.ArgumentParser(description="GraphRAG 系统")
    parser.add_argument("-b", "--build", type=str, help="构建索引：指定文件路径或 glob 模式")
    parser.add_argument("-q", "--query", type=str, help="单次查询")
    parser.add_argument("-i", "--interact", action="store_true", help="交互模式")
    parser.add_argument("-s", "--storage", type=str, default="../data/graphrag_index",
                        help="索引存储路径")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top_k_vectors", type=int, default=5)
    parser.add_argument("--top_k_entities", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_neighbors", type=int, default=5)
    parser.add_argument("--vector_weight", type=float, default=0.5)
    parser.add_argument("--graph_weight", type=float, default=0.5)
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")

    args = parser.parse_args()
    return args, parser


def build_index(
    files,
    storage_path: str,
    chunk_size: int = 512,
    overlap: int = 50,
    embedding_model: str = "BAAI/bge-m3"
) -> None:
    """构建 GraphRAG 索引（向量 + 图谱 + 实体 embedding）"""
    print(f"Building GraphRAG index from {len(files)} files...")

    # 创建索引器
    indexer = get_graphrag_indexer()

    # 读取文件并创建 Document 对象
    documents = []
    for file_path in tqdm(files, desc="Loading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={"source": file_path}
        ))

    # Step 1: 分块
    print("\n[Step 1] Chunking documents...")
    chunks = indexer.index_documents(documents)
    print(f"  Generated {len(chunks)} chunks")

    # Step 2: 构建图谱（提取 -> 对齐 -> 建图）
    print("\n[Step 2] Building knowledge graph...")
    indexer.clear_graph()
    graph_result = indexer.build_graph_from_chunks(chunks)
    print(f"  Graph: {graph_result['entities']} entities, {graph_result['relationships']} relationships")

    # Step 3: 实体向量索引
    print("\n[Step 3] Indexing entities...")
    entity_count = indexer.index_entities()
    print(f"  Indexed {entity_count} entities")

    # Step 4: 保存索引
    print(f"\n[Step 4] Saving index to {storage_path}...")
    indexer.save(storage_path)

    print("\n" + "=" * 60)
    print("Index built successfully!")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Entities: {graph_result['entities']}")
    print(f"  Relationships: {graph_result['relationships']}")
    print("=" * 60)


def load_index(storage_path: str):
    """加载 GraphRAG 索引"""
    if not os.path.exists(storage_path):
        raise FileNotFoundError(f"Index not found at {storage_path}")

    indexer = get_graphrag_indexer()
    indexer.load(storage_path)
    return indexer


def generate_answer(query: str, context_docs, llm=None) -> str:
    """根据上下文生成答案"""
    if llm is None:
        llm = get_llm()

    context_text = "\n\n".join(
        f"[{doc.metadata.get('retrieval_type', 'unknown')} Doc {i+1}]: {doc.page_content}"
        for i, doc in enumerate(context_docs)
    )
    prompt = f"""根据以下参考信息回答问题。如果信息不足，请说明不知道。

参考信息:
{context_text}

问题：{query}

回答："""
    return llm.invoke([HumanMessage(content=prompt)]).content


def graphrag_query(
    query: str,
    retriever,
    llm=None,
    max_context_docs: int = 3
) -> dict:
    """GraphRAG 查询"""
    docs = retriever.retrieve(query)
    context_docs = docs[:max_context_docs]
    answer = generate_answer(query, context_docs, llm)

    return {
        "query": query,
        "answer": answer,
        "references": [
            {
                "content": d.page_content[:200] + "..." if len(d.page_content) > 200 else d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "retrieval_type": d.metadata.get("retrieval_type", "unknown"),
                "score": d.metadata.get("score", 0)
            }
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
        build_index(
            files,
            args.storage,
            args.chunk_size,
            args.overlap,
            args.embedding_model
        )
        return

    # 查询/交互模式
    print("Loading index...")
    try:
        indexer = load_index(args.storage)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # 创建检索器
    retriever = get_graphrag_retriever(
        graph_builder=indexer.graph_builder,
        entity_index=indexer._entity_index,
        entity_metadata=indexer._entity_metadata,
        embedding=indexer.embedding
    )

    llm = get_llm(streaming=True)

    # 单次查询
    if args.query:
        result = graphrag_query(
            args.query,
            retriever,
            llm,
            max_context_docs=3
        )
        print(f"\nAnswer: {result['answer']}")
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
                result = graphrag_query(
                    q,
                    retriever,
                    llm,
                    max_context_docs=3
                )
                print(f"\nAnswer: {result['answer']}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
