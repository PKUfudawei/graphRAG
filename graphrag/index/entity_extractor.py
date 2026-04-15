"""
Entity and relation extractor using LangChain.
使用 LangChain 进行实体和关系提取。
"""
import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser

from ..models.graph_models import GraphDocumentWrapper


class EntityExtractor:
    """
    使用 LangChain 从文本中提取实体和关系。

    Args:
        llm: LangChain 语言模型实例
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self._setup_prompt()

    def _setup_prompt(self):
        """设置实体提取的提示模板"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise entity and relation extraction assistant.

STRICT RULES:
- Return ONLY valid JSON. No extra text.
- Do NOT repeat identical edges.
- Do NOT use LaTeX formatting.
- Each (source, target, relation) must appear ONLY once.
- Use CANONICAL relation types (avoid synonyms).
- Prefer general relations over overly specific ones.
- If multiple similar relations exist, merge them into one.

SCHEMA:
- Nodes: {"name": "", "type": ""}
- Edges: {"source": "", "target": "", "relation": ""}

CONSTRAINTS:
- Use lowercase.
- No underscores.
- Keep relation names SHORT (1-3 words).
- Max 2 relations per entity pair.

QUALITY:
- Remove redundant or repetitive relations.
- Avoid listing variations like "sole friend", "friend", "only friend".
- Choose ONE best relation.

OUTPUT:
Return complete JSON only."""),
            ("human", "Text: {text}\n\nReturn the JSON following the schema:\n{{\"nodes\":[{{\"name\":\"\",\"type\":\"\"}}],\"edges\":[{{\"source\":\"\",\"target\":\"\",\"relation\":\"\"}}]}}")
        ])

    def extract(self, text: str, source: Optional[str] = None) -> GraphDocumentWrapper:
        """
        从单个文本中提取实体和关系。

        Args:
            text: 输入文本
            source: 可选的源标识

        Returns:
            GraphDocumentWrapper 实例
        """
        try:
            # 使用 LangChain 链
            chain = self.prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"text": text})
            return GraphDocumentWrapper.from_extraction_result(result, source)
        except Exception as e:
            # 降级处理：尝试从文本中提取 JSON
            return self._extract_with_fallback(text, source)

    def _extract_with_fallback(self, text: str, source: Optional[str] = None) -> GraphDocumentWrapper:
        """降级提取方法"""
        try:
            # 直接调用 LLM
            response = self.llm.invoke(self.prompt.format(text=text))
            result_text = response.content if hasattr(response, 'content') else str(response)

            # 尝试解析 JSON
            try:
                data = json.loads(result_text)
            except json.JSONDecodeError:
                # 尝试提取 JSON 部分
                match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    data = {}

            return GraphDocumentWrapper.from_extraction_result(data, source)
        except Exception as e:
            print(f"\tError in entity extraction: {e}")
            return GraphDocumentWrapper(nodes=[], edges=[], source=source)

    def extract_batch(self, texts: List[str], sources: Optional[List[str]] = None) -> List[GraphDocumentWrapper]:
        """
        批量提取实体和关系。

        Args:
            texts: 文本列表
            sources: 可选的源标识列表

        Returns:
            GraphDocumentWrapper 列表
        """
        if sources is None:
            sources = [None] * len(texts)

        results = []
        for text, source in zip(texts, sources):
            result = self.extract(text, source)
            results.append(result)

        return results


class ParallelEntityExtractor:
    """
    并行实体提取器。

    Args:
        llm: LangChain 语言模型实例
        max_workers: 最大并发数
    """

    def __init__(self, llm: BaseLanguageModel, max_workers: int = 16):
        self.extractor = EntityExtractor(llm)
        self.max_workers = max_workers

    def extract_parallel(self, texts: List[str], sources: Optional[List[str]] = None) -> List[GraphDocumentWrapper]:
        """
        并行提取实体和关系。

        Args:
            texts: 文本列表
            sources: 可选的源标识列表

        Returns:
            GraphDocumentWrapper 列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        if sources is None:
            sources = [None] * len(texts)

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.extractor.extract, text, source)
                for text, source in zip(texts, sources)
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Extracting entities and relations"
            ):
                results.append(future.result())

        return results
