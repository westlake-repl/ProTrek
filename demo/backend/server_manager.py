import sys

root_dir = __file__.rsplit("/", 3)[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)

import uvicorn
import os
import requests
import time
import socket

from fastapi import FastAPI


app = FastAPI()
base_dir = os.path.dirname(__file__)

# Map the function name to the server directory
FUNCTION_MAP = {
    "search": f"{base_dir}/servers/retrieval/server_list",
    "compute": f"{base_dir}/servers/retrieval/server_list",
    "generate_embedding": f"{base_dir}/servers/embedding_generation/server_list",
}


# Get the IP address of the server
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]

    finally:
        s.close()
        return ip


# Check whether a server is active
def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)

    try:
        result = sock.connect_ex((ip, port))
        if result == 0:
            return True
        else:
            return False

    except Exception as e:
        print(e)
        return False

    finally:
        sock.close()


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


if __name__ == "__main__":
    uvicorn.run("server_manager:app", host="0.0.0.0", port=7861)
