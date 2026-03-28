from concurrent.futures import ThreadPoolExecutor, as_completed
import time, json, faiss
from io import StringIO
from functools import lru_cache


class GlobalSearcher:
    def __init__(self, LLM, embed_model, community_reports, community_embeddings,
                 max_workers=8, batch_size=5):
        self.LLM = LLM
        self.embed_model = embed_model
        self.reports = community_reports
        self.community_embeddings = community_embeddings
        self.max_workers = max_workers
        self.batch_size = batch_size


    @lru_cache(maxsize=128)
    def get_query_embedding(self, query_str):
        emb = self.embed_model.encode([query_str])
        faiss.normalize_L2(emb)
        return emb


    def retrieve_communities(self, query, top_k=10):
        start = time.time()
        query_embedding = self.get_query_embedding(query if isinstance(query, str) else query[0])
        similarities, idx = self.community_embeddings.search(query_embedding, top_k)
        communities = [self.reports[i] for i in idx[0]]
        print(f"[Timer] retrieve_communities: {time.time() - start:.3f}s")
        return communities


    def build_batch_prompt(self, query, batch):
        summaries = "\n\n".join([f"{i+1}. {c['summary']}" for i, c in enumerate(batch)])
        return f"""
You are analyzing a dataset.

Query:
{query}

Community Summaries:
{summaries}

Extract insights relevant to the query from EACH community.

Score definition:
10 = directly answers the query
7-9 = strongly related
4-6 = somewhat related
1-3 = weakly related

Return ONLY valid JSON ARRAY, with each item having:
{{
"insight": "...",
"explanation": "...",
"score": integer 1-10
}}
"""


    def _map_single_batch(self, query, batch):
        system_prompt = """
You are an information extraction system.
Rules:
- Extract insights relevant to the query from all communities.
- Output ONLY valid JSON.
- Do not include explanations outside JSON.
"""
        prompt = self.build_batch_prompt(query, batch)
        result = self.LLM.generate_response(system_prompt=system_prompt, user_prompt=prompt)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []


    def map_step(self, query, communities):
        start_total = time.time()
        insights = []
        batches = [communities[i:i+self.batch_size] for i in range(0, len(communities), self.batch_size)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(self._map_single_batch, query, batch): batch for batch in batches}
            for i, future in enumerate(as_completed(future_to_batch), 1):
                batch_insights = future.result()
                insights.extend(batch_insights)
                print(f"[Timer] map_step batch {i}: {time.time() - start_total:.3f}s")
        print(f"[Timer] map_step total: {time.time() - start_total:.3f}s")
        return insights


    def rank_insights(self, insights, top_k=20):
        start = time.time()
        ranked = sorted(insights, key=lambda x: x["score"], reverse=True)[:top_k]
        print(f"[Timer] rank_insights: {time.time() - start:.3f}s")
        return ranked


    def truncate_insights(self, insights, max_chars=4000):
        buf = StringIO()
        for ins in insights:
            line = f"- [score: {ins['score']}] {ins['insight']}\n"
            if buf.tell() + len(line) > max_chars:
                break
            buf.write(line)
        return buf.getvalue()


    def build_reduce_prompt(self, query, insights_text):
        return f"""
    You are synthesizing insights from a dataset.

    Query:
    {query}

    Insights (with score):
    {insights_text}

    Combine the insights to answer the query.

    Requirements:
    - merge duplicates
    - highlight key themes
    - prioritize higher-score insights
    - provide a structured answer
    """


    def reduce_step(self, query, insights, max_chars=4000):
        start = time.time()
        insights_text = self.truncate_insights(insights, max_chars=max_chars)
        prompt = self.build_reduce_prompt(query, insights_text)
        answer = self.LLM.generate_response(prompt)
        print(f"[Timer] reduce_step: {time.time() - start:.3f}s")
        return answer


    def search(self, query):
        start = time.time()
        communities = self.retrieve_communities(query)
        insights = self.map_step(query, communities)
        insights = self.rank_insights(insights)
        answer = self.reduce_step(query, insights)
        print(f"[Timer] search total: {time.time() - start:.3f}s")
        return answer