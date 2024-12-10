import sys

ROOT_DIR = __file__.rsplit("/", 5)[0]
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import uvicorn
import socket
import os
import torch
import glob

from easydict import EasyDict
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from fastapi import FastAPI

app = FastAPI()
BASE_DIR = os.path.dirname(__file__)
os.makedirs(f"{BASE_DIR}/server_list", exist_ok=True)


@app.get("/generate_embedding")
def generate_embedding(input: str, input_type: str):
    """
    This function is used for generating embeddings
    Args:
        input: Input query

        input_type: Type of input, e.g., "sequence", "structure", "text"
    """
    try:
        # Set server state to busy
        set_state("busy")
        input_modality = input_type.replace("sequence", "protein")
        with torch.no_grad():
            input_embedding = getattr(model, f"get_{input_modality}_repr")([input]).cpu().numpy().tolist()

        return_dict = {
            "input_embedding": input_embedding,
            "temperature": model.temperature.item()
        }

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

model_dir = f"{ROOT_DIR}/weights/ProTrek_650M_UniRef50"

# Load model
model_config = {
    "protein_config": glob.glob(f"{model_dir}/esm2_*")[0],
    "text_config": f"{model_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": glob.glob(f"{model_dir}/foldseek_*")[0],
    "load_protein_pretrained": False,
    "load_text_pretrained": False,
    "from_checkpoint": glob.glob(f"{model_dir}/*.pt")[0]
}

model = ProTrekTrimodalModel(**model_config)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


if __name__ == "__main__":
    # Generate IP flag
    set_state("idle")

    uvicorn.run("server:app", host="0.0.0.0", port=PORT)
