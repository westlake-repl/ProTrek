import sys

root_dir = __file__.rsplit("/", 3)[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)

import uvicorn
import socket
import os
import torch
import pandas as pd

from init_model import model, all_index, valid_subsections
from fastapi import FastAPI
from tqdm import tqdm
from Bio.Align import PairwiseAligner

app = FastAPI()
BASE_DIR = os.path.dirname(__file__)


# Calculate protein sequence identity
def calc_seq_identity(seq1: str, seq2: str) -> float:
    aligner = PairwiseAligner()
    aligner.mode = "local"

    alignment = next(aligner.align(seq1, seq2))
    a1, a2 = alignment
    identity = sum(1 for a, b in zip(a1, a2) if a == b) / len(a1)
    return identity


@app.get("/search_api")
def search(input: str, topk: int, input_type: str, query_type: str, subsection_type: str, db: str):
    """
    This function is used for multi-modal search
    Args:
        input: Input query

        topk: Number of results to return

        input_type: Type of input, e.g., "sequence", "structure", "text"

        query_type: Type of database to search, e.g., "sequence", "structure", "text"

        subsection_type: If db_type is text, search in this subsection

        db: Database name for a specific db_type, e.g., "uniprot", "pdb" in sequence databases

    Returns:

    """
    try:
        # Set server state to busy
        set_state("busy")

        input_modality = input_type.replace("sequence", "protein")
        with torch.no_grad():
            input_embedding = getattr(model, f"get_{input_modality}_repr")([input]).cpu().numpy()

        if query_type == "text":
            index = all_index["text"][db][subsection_type]["index"]
            ids = all_index["text"][db][subsection_type]["ids"]

        else:
            index = all_index[query_type][db]["index"]
            ids = all_index[query_type][db]["ids"]

        if hasattr(index.index_list[0], "nprobe"):
            # index.nprobe = nprobe
            max_num = max(topk, index.nprobe * 256)
        else:
            max_num = index.ntotal

        results, all_scores = index.search(input_embedding, topk, max_num)

        for item in results:
            item[1] /= model.temperature.item()

        all_scores /= model.temperature.item()

        # Retrieve ids based on rank
        topk_ids = []
        for i in tqdm(range(topk)):
            index_rk, score, rank = results[i]
            topk_ids.append(ids[index_rk].get(rank))

        return_dict = {
            "results": results,
            "all_scores": all_scores.tolist(),
            "ids": topk_ids
        }

        results = return_dict["results"]
        ids = return_dict["ids"]

        # If both the input and output are protein sequences, calculate the sequence identity
        if input_type == "sequence" and query_type == "sequence":
            seq_identities = []
            for i in range(topk):
                hit_seq = ids[i].split("\t")[1]
                seq_identities.append(calc_seq_identity(input, hit_seq))

            seq_identities = [f"{identity * 100:.2f}%" for identity in seq_identities]

        # Get topk ids
        topk_ids = []
        topk_scores = []
        topk_seqs = []
        topk_lengths = []
        for i in range(topk):
            index_rk, score, rank = results[i]
            now_id = ids[i].split("\t")[0].replace("|", "\\|")
            if query_type != "text":
                now_id = now_id[:20] + "..." if len(now_id) > 20 else now_id
            topk_scores.append(score)

            if query_type == "text":
                topk_ids.append(now_id)
            else:
                if db in ["UniRef50", "Uncharacterized", "Swiss-Prot"]:
                    # Provide link to uniprot website
                    topk_ids.append(f"[{now_id}](https://www.uniprot.org/uniprotkb/{now_id})")
                elif db == "PDB":
                    # Provide link to pdb website
                    pdb_id = now_id.split("-")[0]
                    topk_ids.append(f"[{now_id}](https://www.rcsb.org/structure/{pdb_id})")
                else:
                    topk_ids.append(now_id)

            if query_type != "text":
                _, ori_seq, ori_len = ids[i].split("\t")
                topk_seqs.append(ori_seq)
                topk_lengths.append(ori_len)

        limit = 1000
        if query_type == "text":
            df = pd.DataFrame({"Id": topk_ids[:limit], "Matching score": topk_scores[:limit]})
            if len(topk_ids) > limit:
                info_df = pd.DataFrame({"Id": ["Download the file to check all results"], "Matching score": ["..."]},
                                       index=[1000])
                df = pd.concat([df, info_df], axis=0)

        elif input_type == "sequence" and query_type == "sequence":
            df = pd.DataFrame({"Id": topk_ids[:limit], "Sequence": topk_seqs[:limit],
                               "Length": topk_lengths[:limit], "Sequence identity": seq_identities[:limit],
                               "Matching score": topk_scores[:limit]})
            if len(topk_ids) > limit:
                info_df = pd.DataFrame(
                    {"Id": ["Download the file to check all results"], "Sequence": ["..."], "Length": ["..."],
                     "Sequence identity": ["..."], "Matching score": ["..."]}, index=[1000])
                df = pd.concat([df, info_df], axis=0)

        else:
            df = pd.DataFrame({"Id": topk_ids[:limit], "Sequence": topk_seqs[:limit], "Length": topk_lengths[:limit],
                               "Matching score": topk_scores[:limit]})
            if len(topk_ids) > limit:
                info_df = pd.DataFrame(
                    {"Id": ["Download the file to check all results"], "Sequence": ["..."], "Length": ["..."],
                     "Matching score": ["..."]},
                    index=[1000])
                df = pd.concat([df, info_df], axis=0)

        return_dict = {"df": df.to_dict(orient="records")}

    except Exception as e:
        return_dict = {"error": str(e)}

    finally:
        # Set server state to idle
        set_state("idle")

    return return_dict


# Set server state
def set_state(state: str):
    flag_path = f"{BASE_DIR}/cache/{get_ip()}:{PORT}.flag"
    with open(flag_path, "w") as w:
        w.write(state)


# Get the IP address of the server
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]

    finally:
        s.close()
        return ip


# Check whether a port is in use
def check_port_in_use(port, host='127.0.0.1'):
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, int(port)))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()


PORT = 7862
while check_port_in_use(PORT):
    PORT += 1

if __name__ == "__main__":
    # Generate IP flag
    set_state("idle")

    uvicorn.run("test:app", host="0.0.0.0", port=PORT)
