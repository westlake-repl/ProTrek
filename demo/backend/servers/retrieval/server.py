import sys

root_dir = __file__.rsplit("/", 5)[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)

import uvicorn
import os
import torch
import json
import requests
import numpy as np
import argparse

from init_index import init_index
from fastapi import FastAPI
from tqdm import tqdm
from utils.server_tool import check_port_in_use, get_ip


app = FastAPI()
BASE_DIR = os.path.dirname(__file__)
os.makedirs(f"{BASE_DIR}/cache", exist_ok=True)
os.makedirs(f"{BASE_DIR}/server_list", exist_ok=True)


@app.get("/search")
def search(manager_ip_port: str,
           input: str,
           topk: int,
           input_type: str,
           query_type: str,
           subsection_type: str,
           db: str):
    """
    This function is used for multi-modal search
    Args:
        manager_ip_port: IP and port of the server manager

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

        # Get input embedding
        params = {
            "input": input,
            "input_type": input_type,
        }

        url = f"http://{manager_ip_port}/generate_embedding"
        response = requests.get(url=url, params=params).json()
        input_embedding = np.array(response["input_embedding"])
        temperature = response["temperature"]

        if query_type == "text":
            index = all_index["text"][db][subsection_type]["index"]
            ids = all_index["text"][db][subsection_type]["ids"]

        else:
            index = all_index[query_type][db]["index"]
            ids = all_index[query_type][db]["ids"]
        
        # Set topk
        topk = min(topk, index.ntotal)
        
        if hasattr(index.index_list[0], "nprobe"):
            # index.nprobe = nprobe
            max_num = max(topk, index.nprobe * 256)
        else:
            max_num = index.ntotal

        results, all_scores = index.search(input_embedding, topk, max_num)

        for item in results:
            item[1] /= temperature

        all_scores /= temperature
        
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

        file_path = f"{BASE_DIR}/cache/{get_ip()}:{PORT}.json"
        with open(file_path, "w") as w:
            json.dump(return_dict, w)
            return_dict = {"file_path": file_path}

    except Exception as e:
        return_dict = {"error": str(e)}
        print(e)
        # raise e

    finally:
        # Set server state to idle
        set_state("idle")

    return return_dict


@app.get("/compute")
def compute_score(manager_ip_port: str, input_type_1: str, input_1: str, input_type_2: str, input_2: str):
    """
    This function is used to compute the similarity score between two inputs
    Args:
        manager_ip_port: IP and port of the server manager

        input_type_1: Type of input 1, e.g., "sequence", "structure", "text"

        input_1: Input query 1

        input_type_2: Type of input 2, e.g., "sequence", "structure", "text"

        input_2: Input query 2
    """
    try:
        # Set server state to busy
        set_state("busy")

        with torch.no_grad():
            input_reprs = []
            for input_type, input in [(input_type_1, input_1), (input_type_2, input_2)]:
                # Get input embedding
                params = {
                    "input": input,
                    "input_type": input_type,
                }

                url = f"http://{manager_ip_port}/generate_embedding"
                response = requests.get(url=url, params=params).json()
                input_embedding = np.array(response["input_embedding"])
                input_reprs.append(input_embedding)

                temperature = response["temperature"]

            score = input_reprs[0] @ input_reprs[1].T / temperature
            return_dict = {"score": f"{score.item():.4f}"}

    except Exception as e:
        return_dict = {"error": str(e)}

    finally:
        # Set server state to idle
        set_state("idle")

    return return_dict


# Set server state
def set_state(state: str):
    flag_path = f"{BASE_DIR}/server_list/{get_ip()}:{PORT}.flag"
    with open(flag_path, "w") as w:
        state_dict = {
            "state": state,
            "sequence": list(all_index["sequence"].keys()),
            "structure": list(all_index["structure"].keys()),
            "text": list(all_index["text"].keys()),
        }
        json.dump(state_dict, w, indent=4)


# Specify the server port
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--sequence_db', type=str, default=None,
                    help="Specify which sequence databases to use. If None, follow the config file.")
parser.add_argument('--structure_db', type=str, default=None,
                    help="Specify which structure databases to use. If None, follow the config file.")
parser.add_argument('--text_db', type=str, default=None,
                    help="Specify which text databases to use. If None, follow the config file.")
args = parser.parse_args()

if args.port is None:
    PORT = 7862
    while check_port_in_use(PORT):
        PORT += 1

else:
    PORT = args.port
    

# Initialize the index
sequence_db = args.sequence_db.split(",") if args.sequence_db else None
structure_db = args.structure_db.split(",") if args.structure_db else None
text_db = args.text_db.split(",") if args.text_db else None
all_index, valid_subsections = init_index(sequence_db, structure_db, text_db)


if __name__ == "__main__":
    # Generate IP flag
    set_state("idle")
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)

