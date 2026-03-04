import tiktoken, json
from tqdm import tqdm


class Chunk:
    def __init__(self, chunk_id=-1, chunk_text="", source=None, start_token=0, end_token=0):
        self.id = chunk_id
        self.text = chunk_text
        self.source = source 
        self.start_token = start_token
        self.end_token = end_token 
        self.token_count = end_token - start_token
        self.text_count = len(chunk_text)

    def __repr__(self):
        return '\n'.join([
            "Chunk",
            f"=> id: {self.id}, source: {self.source}, token_range: [{self.start_token}, {self.end_token}), token_count: {self.token_count}, text_count: {self.text_count}",
            f"=> text:",
            f"{self.text}"
        ])


class Chunker:
    def __init__(self, chunk_size=512, overlap=50, encoding_model="cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding(encoding_model)


    def chunk_text(self, text: str, source=None):
        """
        Token-based chunking (GraphRAG style)
        """
        tokens = self.encoder.encode(text)
        chunks = []
        total_tokens = len(tokens)

        for idx, start in tqdm(enumerate(range(0, total_tokens, self.chunk_size - self.overlap)), desc=f"Chunking text from source: {source if source else 'unknown'}"):
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(Chunk(
                chunk_id=idx, chunk_text=chunk_text, source=source, 
                start_token=start, end_token=end
            ))

        return chunks


    def chunk_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.chunk_text(text, source=file_path)
    
    
    def save_chunks(self, file_path, chunks=None):
        chunks_data = [{
            'id': chunk.id, 'text': chunk.text, 'start_token': chunk.start_token, 'end_token': chunk.end_token, 
            'token_count': chunk.token_count, 'text_count': chunk.text_count, 'source': chunk.source
        } for chunk in (self.chunks if chunks is None else chunks)]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {file_path}")
    
    
    def load_chunks(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        chunks = [Chunk(
            chunk_id=data['id'], chunk_text=data['text'], source=data.get('source', None), 
            start_token=data.get('start_token', 0), end_token=data.get('end_token', 0)
        ) for data in chunks_data]
        print(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks