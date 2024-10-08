import faiss
import numpy as np

from typing import Union, List
from tqdm import tqdm


class FaissIndex:
    """
    Wrapper for Faiss index
    """
    def __init__(self,
                 index_path: Union[str, List[str]],
                 IO_FLAG_MMAP: bool = True,
                 nprobe: int = 1):
        """
        Initialize Faiss index
        Args:
            index_path: Path to index file
            IO_FLAG_MMAP: Whether to use mmap
            nprobe: Number of probes
        """
        if isinstance(index_path, str):
            index_path = [index_path]

        self.index_list = []
        self.ntotal = 0
        self.nprobe = nprobe
        for path in index_path:
            if IO_FLAG_MMAP:
                index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
            else:
                index = faiss.read_index(path)

            index.metric_type = faiss.METRIC_INNER_PRODUCT
            
            self.ntotal += index.ntotal
            self.index_list.append(index)

    def search(self, query, k, max_num):
        """
        Search for top k similar items
        Args:
            query: Query vector
            k:  Number of similar items
            max_num: Maximum number of items to be returned
        
        Returns:
            results: List of tuples (index, score, rank) for top k results
            all_scores: All scores for further analysis
        """
        results = []
        all_scores = []
        for index_rk, index in enumerate(tqdm(self.index_list, "Searching...")):
            if hasattr(index, "nprobe"):
                index.nprobe = self.nprobe
                
            scores, ranks = index.search(query, max_num)
            scores, ranks = scores[0], ranks[0]
            
            # Remove inf values
            selector = scores > -1
            scores = scores[selector]
            ranks = ranks[selector]
            
            all_scores += scores.tolist()
            
            for i in range(k):
                score = scores[i]
                rk = ranks[i]
                results.append([index_rk, score, rk])
            
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:k], np.array(all_scores)
    