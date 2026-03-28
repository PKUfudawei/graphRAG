import tiktoken, json
import numpy as np
import faiss, os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class Chunk:
    def __init__(self, chunk_id=-1, chunk_text="", source=None, start_token=0, end_token=0, embedding=None):
        self.id = chunk_id
        self.text = chunk_text
        self.source = source
        self.start_token = start_token
        self.end_token = end_token 
        self.token_count = end_token - start_token
        self.text_count = len(chunk_text)
        self.embedding = embedding


    def __repr__(self):
        return f"Chunk(id={self.id}, tokens={self.token_count}, source={self.source})\nMore attributes hidden for brevity: start_token, end_token, text_count, embedding"


class Chunker:
    def __init__(self, chunk_size, overlap, encoding_model="cl100k_base", embed_model=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding(encoding_model)
        self.embed_model = embed_model


    def chunk_text(self, text: str, source=None):
        tokens = self.encoder.encode(text)
        total_tokens = len(tokens)
        chunks = []

        for idx, start in enumerate(range(0, total_tokens, self.chunk_size - self.overlap)):
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(Chunk(
                chunk_id=idx, chunk_text=chunk_text, source=source, 
                start_token=start, end_token=end
            ))

        return chunks


    def embed_chunks(self, chunks, batch_size=64):
        if self.embed_model is None:
            print("No embedding model specified, skipping embedding step")
            return chunks

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks


    def save_chunks(self, chunks, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        chunks_data = [{
            'id': chunk.id, 'text': chunk.text, 'start_token': chunk.start_token, 'end_token': chunk.end_token, 
            'token_count': chunk.token_count, 'text_count': chunk.text_count, 'source': chunk.source,
        } for chunk in chunks]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"\tSaved {len(chunks)} chunks to {path}")
    
    
    def load_chunks(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        chunks = [Chunk(
            chunk_id=data['id'], chunk_text=data['text'], source=data.get('source', None), 
            start_token=data.get('start_token', 0), end_token=data.get('end_token', 0),
        ) for data in chunks_data]
        print(f"Loaded {len(chunks)} chunks from {path}")
        return chunks


    def build_index(self, chunks, batch_size=10_000):
        index = faiss.IndexFlatIP(chunks[0].embedding.shape[0])
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            embeddings = np.vstack([chunk.embedding for chunk in batch_chunks]).astype("float32")
            index.add(embeddings)

        return index


    def save_index(self, index, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        faiss.write_index(index, path)
        print(f"\tChunks index saved to {path}")


    def load_index(self, path):
        index = faiss.read_index(path)
        print(f"\tChunks index loaded from {path}")
        return index

 
    def chunk_file(self, path, chunks_path=None, index_path=None, embed_batch=64, index_batch=10_000):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self.chunk_text(text, source=path)
        chunks = self.embed_chunks(chunks, batch_size=embed_batch)

        if chunks_path is not None:
            self.save_chunks(chunks, chunks_path)
        if index_path is not None:
            faiss_index = self.build_index(chunks, batch_size=index_batch)
            self.save_index(faiss_index, index_path)

        return chunks


    def chunk_files(self, files, chunks_path=None, index_path=None, 
                    embed_batch=64, index_batch=10_000, max_workers=16):
        all_chunks = []
        
        def process_file(path):
            return self.chunk_file(
                path,
                chunks_path=None,
                index_path=None,
                embed_batch=embed_batch,
                index_batch=index_batch
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in files}
            for future in tqdm(as_completed(futures), total=len(files), desc="Chunking files"):
                file = futures[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

        if chunks_path is not None:
            self.save_chunks(all_chunks, chunks_path)

        if index_path is not None:
            faiss_index = self.build_index(all_chunks, batch_size=index_batch)
            self.save_index(faiss_index, index_path)

        return all_chunks
