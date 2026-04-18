"""
GraphRAG 系统主模块 - 使用 GraphRAGIndexer 和 GraphRAGRetriever
"""

import os
import sys
import argparse
import pickle
from glob import glob
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS

# Add paths
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from graphrag.indexer import get_graphrag_indexer
from graphrag.retriever import get_graphrag_retriever
from models.llm import get_llm
from models.embedding import get_embedding


def parse_arguments():
    parser = argparse.ArgumentParser(description="GraphRAG 系统")
    parser.add_argument("-b", "--build", type=str, help="构建索引：指定文件路径或 glob 模式")
    parser.add_argument("-q", "--query", type=str, help="单次查询")
    parser.add_argument("-i", "--interact", action="store_true", help="交互模式")
    parser.add_argument("-s", "--storage", type=str, default="./storage/graphrag_index",
                        help="索引存储路径")
    parser.add_argument("--incremental", action="store_true", help="增量更新模式")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top_k_vectors", type=int, default=5)
    parser.add_argument("--top_k_entities", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_neighbors", type=int, default=5)
    parser.add_argument("--vector_weight", type=float, default=0.5)
    parser.add_argument("--graph_weight", type=float, default=0.5)
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--max_workers", type=int, default=16, help="并行提取的最大工作线程数")
    parser.add_argument("--enable_thinking", type=str, default="true", help="是否启用 thinking 模式 (true/false，默认 true)")

    args = parser.parse_args()
    return args, parser


def build_index(
    files,
    storage_path: str,
    max_workers: int = 16,
    incremental: bool = False,
    enable_thinking: bool = False
) -> None:
    """构建 GraphRAG 索引（向量 + 图谱 + 实体 embedding）"""
    mode_str = "Incremental update" if incremental else "Full build"
    print(f"{mode_str} from {len(files)} files...")
    print(f"enable_thinking: {enable_thinking}")

    # 读取文件并创建 Document 对象
    documents = []
    for file_path in tqdm(files, desc="Loading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={"source": file_path}
        ))

    # 使用 GraphRAGIndexer 的一站式索引方法
    indexer = get_graphrag_indexer(max_workers=max_workers, enable_thinking=enable_thinking)
    chunks, vectorstore, graph, entity_index = indexer.index_documents(
        documents,
        database_path=storage_path,
        incremental=incremental
    )

    print("\n" + "=" * 60)
    print(f"Index {mode_str.lower()} successfully!")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Entities: {graph.number_of_nodes()}")
    print(f"  Relationships: {graph.number_of_edges()}")
    print("=" * 60)


def load_index(storage_path: str):
    """加载 GraphRAG 索引"""
    # 加载 graph
    graph_path = os.path.join(storage_path, 'graph.pkl')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found at {graph_path}")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    # 加载 entities
    entities_path = os.path.join(storage_path, 'entities.pkl')
    if not os.path.exists(entities_path):
        raise FileNotFoundError(f"Entities not found at {entities_path}")
    with open(entities_path, "rb") as f:
        entities_data = pickle.load(f)

    # 加载 vectorstore
    vectorstore_path = os.path.join(storage_path, 'vectorstore')
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vectorstore not found at {vectorstore_path}")
    embedding = get_embedding()
    # FAISS 需要 Embeddings 对象，使用 embedding.embed_model
    embed_model = embedding.embed_model if hasattr(embedding, 'embed_model') else embedding
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embed_model,
        allow_dangerous_deserialization=True
    )

    return {
        "graph": graph,
        "entity_index": entities_data["index"],
        "entity_metadata": entities_data["metadata"],
        "vectorstore": vectorstore,
        "embedding": embedding
    }


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
        enable_thinking = args.enable_thinking.lower() == "true"
        build_index(
            files,
            args.storage,
            args.max_workers,
            args.incremental,
            enable_thinking
        )
        return

    # 查询/交互模式
    print("Loading index...")
    try:
        index_data = load_index(args.storage)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # 创建检索器
    retriever = get_graphrag_retriever(
        graph=index_data["graph"],
        entity_index=index_data["entity_index"],
        entity_metadata=index_data["entity_metadata"],
        vectorstore=index_data["vectorstore"],
        embedding=index_data["embedding"]
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
