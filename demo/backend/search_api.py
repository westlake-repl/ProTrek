import sys
root_dir = __file__.rsplit("/", 3)[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
import uvicorn
import socket
import os

# from init_model import model, all_index, valid_subsections
from fastapi import FastAPI


app = FastAPI()
BASE_DIR = os.path.dirname(__file__)


@app.get("/search")
def search(input: str, topk: int, input_type: str, db_type: str, subsection_type: str, db: str):
    """
    This function is used for multi-modal search
    Args:
        input: Input query
        
        topk: Number of results to return
        
        input_type: Type of input, e.g., "sequence", "structure", "text"
        
        db_type: Type of database to search, e.g., "sequence", "structure", "text"
        
        subsection_type: If db_type is text, search in this subsection
        
        db: Database name for a specific db_type, e.g., "uniprot", "pdb" in sequence databases

    Returns:

    """
    
    # Set server state to busy
    set_state("busy")
    
    # Handle the request
    import time
    time.sleep(1000)
    
    # Set server state to idle
    set_state("idle")
    
    return {
        "input": input,
        "topk": topk,
        "input_type": input_type,
        "db_type": db_type,
        "subsection_type": subsection_type,
        "db": db,
    }


# Set server state
def set_state(state: str):
    flag_path = f"{BASE_DIR}/server_list/{get_ip()}:7862"
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
    

if __name__ == "__main__":
    # Generate IP flag
    set_state("idle")
    
    uvicorn.run("search_api:app", host="0.0.0.0", port=7862, reload=True)
    