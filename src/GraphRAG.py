class GraphRAG:
    def __init__(self, global_search, local_search):
        self.global_search = global_search
        self.local_search = local_search

    def query(self, query):
        if len(query.split()) > 8:
            return self.global_search.search(query)
        else:
            return self.local_search.search(query)