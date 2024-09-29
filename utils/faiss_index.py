import faiss
import numpy as np

from typing import Union, List
from tqdm import tqdm


class FaissIndex:
    """
    Wrapper for Faiss index
    """
    def __init__(self, index_path: Union[str, List[str]], IO_FLAG_MMAP: bool = True):
        """
        Initialize Faiss index
        Args:
            index_path: Path to index file
        """
        if isinstance(index_path, str):
            index_path = [index_path]

        self.index_list = []
        for path in tqdm(index_path, "Loading Faiss index..."):
            if IO_FLAG_MMAP:
                index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
            else:
                index = faiss.read_index(path)

            index.metric_type = faiss.METRIC_INNER_PRODUCT
            self.index_list.append(index)

    def search(self, query, k):
        """
        Search for top k similar items
        Args:
            query: Query vector
            k:  Number of similar items
        """

        if len(self.index_list) == 1:
            return self.index_list[0].search(query, k)

        else:
            results = []
            for i, index in enumerate(self.index_list):
                scores, ranks = index.search(query, k)
                for score, rk in zip(scores[0], ranks[0]):
                    results.append((i, score, rk))

            results = sorted(results, key=lambda x: x[1], reverse=True)
            return results[:k]