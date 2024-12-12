import faiss
import numpy as np
import glob
import os

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
        for path in tqdm(index_path, "Loading index..."):
            if IO_FLAG_MMAP:
                index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
            else:
                index = faiss.read_index(path)

            index.metric_type = faiss.METRIC_INNER_PRODUCT
            
            # # Additional raw embeddings are required for IVFPQ
            # if isinstance(index, faiss.IndexIVFPQ):
            #     file_dir = os.path.dirname(path)
            #     npy_path = glob.glob(f"{file_dir}/*.npy")
            #     assert len(npy_path) == 1, f"Multiple npy files found in {file_dir}"
            #
            #     npy_path = npy_path[0]
            #     index.raw_embeddings = np.memmap(npy_path, dtype=np.float32, mode="r", shape=(index.ntotal, index.d))
                
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
            
            # # If the index is IVFPQ, we need to calculate the real scores using raw embeddings
            # if isinstance(index, faiss.IndexIVFPQ):
            #     hit_embeddings = np.empty((len(ranks), index.d), dtype=np.float32)
            #     for i, hit_id in enumerate(tqdm(ranks)):
            #         hit_embeddings[i] = index.raw_embeddings[hit_id]
            #         pass
            #     # scores = np.dot(query, hit_embeddings.T).flatten()
            
            all_scores += scores.tolist()
            
            for i in range(min(k, len(ranks))):
                score = scores[i]
                rk = ranks[i]
                results.append([index_rk, score, int(rk)])

        results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
        # # If the index is IVFPQ, we need to calculate the real scores using raw embeddings
        # if isinstance(self.index_list[0], faiss.IndexIVFPQ):
        #     hit_embeddings = np.empty((k, self.index_list[0].d), dtype=np.float32)
        #     for i, (index_rk, _, hit_id) in enumerate(tqdm(results)):
        #         hit_embeddings[i] = self.index_list[index_rk].raw_embeddings[hit_id]
        #
        #     scores = np.dot(query, hit_embeddings.T).flatten()
        #     for i, score in enumerate(scores):
        #         results[i][1] = score
        #
        #     # Sort results again
        #     results = sorted(results, key=lambda x: x[1], reverse=True)

        return results, np.array(all_scores)
    