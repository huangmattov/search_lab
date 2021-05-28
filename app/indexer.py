import faiss
import tempfile
import numpy as np

from .utils.logging_helper import LoggingHelper


class FaissIndexer:
    LOGGER = LoggingHelper("FaissIndexer").logger
    """
    Builds an index using the Faiss library.
    """

    def __init__(self, path: str = None, threshold: float = 0.3):
        self.model = None
        self.threshold = threshold
        self.names = None
        if path:
            self.load(path)
            FaissIndexer.LOGGER.info("Saved indexer retrieved from [{}]".format(path))

    def load(self, path: str):
        # Load index
        self.model = faiss.read_index(path)

    def load_bytes(self, byte_data):
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as f:
            f.write(byte_data)
        self.model = faiss.read_index(fname)

    def create_model(self, dim):
        # Create embeddings index. Inner product is equal to cosine similarity on normalized vectors.
        self.model = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)

    def index(self, embeddings: np.ndarray):
        self.model.add(embeddings)

    def search(self, query: np.ndarray, limit: int):
        # Run the query
        scores, ids = self.model.search(query.reshape(1, -1), limit)
        # Map results to [(id, score)]
        return list(zip(ids[0].tolist(), [max(1 - score, 0) for score in scores[0].tolist()]))

    def batch_search(self, queries: np.ndarray, limit: int):
        # check if batch search is possible
        results = []
        for query in queries:
            results.append(self.search(query, limit))

        return results

    def save(self, path: str):
        # Write index
        faiss.write_index(self.model, path)

    def save_bytes(self):
        _, fname = tempfile.mkstemp()
        faiss.write_index(self.model, fname)
        return fname


