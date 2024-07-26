import faiss
import numpy as np
import pandas as pd
import os
import yaml

from easydict import EasyDict
from utils.constants import sequence_level
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from tqdm import tqdm


def load_model():
    model_config = {
        "protein_config": f"{config.model_dir}/esm2_t33_650M_UR50D",
        "text_config": f"{config.model_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": f"{config.model_dir}/foldseek_t30_150M",
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": f"{config.model_dir}/ProTrek_650M_UniRef50.pt"
    }

    model = ProTrekTrimodalModel(**model_config)
    model.eval()
    return model


def load_faiss_index(index_path: str):
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
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
        sequence_index = load_faiss_index(index_path)

        id_path = f"{index_dir}/ids.tsv"
        uniprot_ids = pd.read_csv(id_path, sep="\t", header=None).values.flatten()

        all_index["sequence"][db_name] = {"index": sequence_index, "ids": uniprot_ids}

    # Load protein structure index
    print("Loading structure index...")
    all_index["structure"] = {}
    for db in tqdm(config.structure_index_dir, desc="Loading structure index..."):
        db_name = db["name"]
        index_dir = db["index_dir"]

        index_path = f"{index_dir}/structure.index"
        structure_index = load_faiss_index(index_path)

        id_path = f"{index_dir}/ids.tsv"
        uniprot_ids = pd.read_csv(id_path, sep="\t", header=None).values.flatten()

        all_index["structure"][db_name] = {"index": structure_index, "ids": uniprot_ids}
    
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
                
            text_index = load_faiss_index(index_path)
            
            id_path = f"{text_dir}/{subsection.replace(' ', '_')}_ids.tsv"
            text_ids = pd.read_csv(id_path, sep="\t", header=None).values.flatten()
            
            all_index["text"][db_name][subsection] = {"index": text_index, "ids": text_ids}
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

device = "cuda"

print(config)

print("Loading model...")
model = load_model()
model.to(device)

all_index, valid_subsections = load_index()
print("Done...")
# model = None
# all_index, valid_subsections = {"text": {}}, {}