import sys

ROOT_DIR = __file__.rsplit("/", 3)[0]
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import uvicorn
import os
import requests

from utils.server_tool import get_ip, check_port
from fastapi import FastAPI


app = FastAPI()
BACKEND_DIR = f"{ROOT_DIR}/demo/backend"

# Map the function name to the server directory
FUNCTION_MAP = {
    "search": f"{BACKEND_DIR}/servers/retrieval/server_list",
    "compute": f"{BACKEND_DIR}/servers/retrieval/server_list",
    "generate_embedding": f"{BACKEND_DIR}/servers/embedding_generation/server_list",
}


def get_idle_node(server_dir: str) -> str:
    """
    Find an idle node in the server list to perform the request
    
    Returns:
        ip_port: IP address of the idle node
    """

    # Find the first idle node
    server_list = list(filter(lambda x: x.endswith(".flag"), os.listdir(server_dir)))
    # Sort by the last modified time
    server_list.sort(key=lambda file: os.path.getmtime(f"{server_dir}/{file}"))
    
    for ip_port in server_list:
        ip, port = ip_port.split(".flag")[0].split(":")
        ip_info = f"{server_dir}/{ip_port}"

        # Remove inaccessible server
        if not check_port(ip, int(port)):
            os.remove(ip_info)
            continue

        with open(ip_info, "r") as r:
            state = r.read()

        if state == "idle":
            return ip_port.split(".flag")[0]
    
    # No idle node
    raise Exception("No idle node available")
        

@app.get("/search")
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
    ip = get_idle_node(FUNCTION_MAP["search"])
    print(ip)

    # Send request to the idle node
    url = f"http://{ip}/search"
    params = {
        "manager_ip_port": f"{get_ip()}:7861",
        "input": input,
        "topk": topk,
        "input_type": input_type,
        "query_type": query_type,
        "subsection_type": subsection_type,
        "db": db,
    }

    response = requests.get(url=url, params=params).json()
    return response


@app.get("/compute")
def compute_score(input_type_1: str, input_1: str, input_type_2: str, input_2: str):
    """
    This function is used to compute the similarity score between two inputs
    Args:
        input_type_1: Type of input 1, e.g., "sequence", "structure", "text"

        input_1: Input query 1

        input_type_2: Type of input 2, e.g., "sequence", "structure", "text"

        input_2: Input query 2
    """
    ip = get_idle_node(FUNCTION_MAP["compute"])
    
    # Send request to the idle node
    url = f"http://{ip}/compute"
    params = {
        "manager_ip_port": f"{get_ip()}:7861",
        "input_type_1": input_type_1,
        "input_1": input_1,
        "input_type_2": input_type_2,
        "input_2": input_2,
    }
    
    response = requests.get(url=url, params=params).json()
    return response


@app.get("/generate_embedding")
def generate_embedding(input: str, input_type: str):
    """
    This function is used for generating embeddings
    Args:
        input: Input query

        input_type: Type of input, e.g., "sequence", "structure", "text"
    """
    ip = get_idle_node(FUNCTION_MAP["generate_embedding"])

    # Send request to the idle node
    url = f"http://{ip}/generate_embedding"
    params = {
        "input": input,
        "input_type": input_type,
    }

    response = requests.get(url=url, params=params).json()
    return response


PORT = 7861


if __name__ == "__main__":
    uvicorn.run("server_manager:app", host="0.0.0.0", port=7861)
