import faiss
import numpy as np
import pandas as pd
import os
import yaml
import glob
import torch

from easydict import EasyDict
from utils.constants import sequence_level
from utils.file_reader import FileReader
from utils.faiss_index import FaissIndex
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from tqdm import tqdm


def load_model():
    model_config = {
        "protein_config": glob.glob(f"{config.model_dir}/esm2_*")[0],
        "text_config": f"{config.model_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": glob.glob(f"{config.model_dir}/foldseek_*")[0],
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": glob.glob(f"{config.model_dir}/*.pt")[0]
    }

    model = ProTrekTrimodalModel(**model_config)
    model.eval()
    return model


def load_faiss_index(index_path: str):
    if config.faiss_config.IO_FLAG_MMAP:
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(index_path)
        
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    return index


def load_index():
    all_index = {}
    
    # Load protein sequence index
    all_index["sequence"] = {}
    for db in tqdm(config.sequence_index_dir, desc="Loading sequence index..."):
        db_name = db["name"]
        index_dir = db["index_dir"]

        index_path = f"{index_dir}/sequence.index"
        if os.path.exists(index_path):
            sequence_index = FaissIndex(index_path, nprobe=db.get("nprobe", 1))

            id_path = f"{index_dir}/ids.tsv"
            ids = FileReader(id_path)

            all_index["sequence"][db_name] = {"index": sequence_index, "ids": [ids]}

        else:
            # The index contains multiple files
            index_files = []
            ids = []
            for dir_name in os.listdir(index_dir):
                if not os.path.isdir(f"{index_dir}/{dir_name}"):
                    continue
                    
                index_path = f"{index_dir}/{dir_name}/sequence.index"
                index_files.append(index_path)

                id_path = f"{index_dir}/{dir_name}/ids.tsv"
                ids.append(FileReader(id_path))

            sequence_index = FaissIndex(index_files, nprobe=db.get("nprobe", 1))
            all_index["sequence"][db_name] = {"index": sequence_index, "ids": ids}

    # Load protein structure index
    print("Loading structure index...")
    all_index["structure"] = {}
    for db in tqdm(config.structure_index_dir, desc="Loading structure index..."):
        db_name = db["name"]
        index_dir = db["index_dir"]

        index_path = f"{index_dir}/structure.index"
        structure_index = FaissIndex(index_path)

        id_path = f"{index_dir}/ids.tsv"
        ids = FileReader(id_path)

        all_index["structure"][db_name] = {"index": structure_index, "ids": [ids]}
    
    # Load text index
    all_index["text"] = {}
    valid_subsections = {}
    for db in tqdm(config.text_index_dir, desc="Loading text index..."):
        db_name = db["name"]
        index_dir = db["index_dir"]
        all_index["text"][db_name] = {}
        text_dir = f"{index_dir}/subsections"
        
        # Remove "Taxonomic lineage" from sequence_level. This is a special case which we don't need to index.
        valid_subsections[db_name] = set()
        sequence_level.add("Global")
        for subsection in tqdm(sequence_level):
            index_path = f"{text_dir}/{subsection.replace(' ', '_')}.index"
            if not os.path.exists(index_path):
                continue

            text_index = FaissIndex(index_path)
            
            id_path = f"{text_dir}/{subsection.replace(' ', '_')}_ids.tsv"
            text_ids = FileReader(id_path)
            
            all_index["text"][db_name][subsection] = {"index": text_index, "ids": [text_ids]}
            valid_subsections[db_name].add(subsection)
    
    # Sort valid_subsections
    for db_name in valid_subsections:
        valid_subsections[db_name] = sorted(list(valid_subsections[db_name]))

    return all_index, valid_subsections


# Load the config file
root_dir = __file__.rsplit("/", 3)[0]
config_path = f"{root_dir}/demo/config.yaml"
with open(config_path, 'r', encoding='utf-8') as r:
    config = EasyDict(yaml.safe_load(r))

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = load_model()
model.to(device)

all_index, valid_subsections = load_index()
print("Done...")
# model = None
# all_index, valid_subsections = {"text": {}, "sequence": {"UniRef50": None}, "structure": {"UniRef50": None}}, {}