import faiss
import pandas as pd
import os

from utils.constants import sequence_level
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel


def load_model():
    config = {
        "protein_config": "weights/ProTrek_650M_UniRef50/esm2_t33_650M_UR50D",
        "text_config": "weights/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": "weights/ProTrek_650M_UniRef50/foldseek_t30_150M",
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": "weights/ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt"
    }

    model = ProTrekTrimodalModel(**config)
    model.eval()
    return model


def load_index():
    index_dir = "weights/faiss_index/faiss_index_ProTrek_650M_UniRef50"
    all_index = {}
    
    # Load protein sequence index
    index_path = f"{index_dir}/sequence.index"
    sequence_index = faiss.read_index(index_path)
    
    id_path = f"{index_dir}/sequence_ids.tsv"
    uniprot_ids = pd.read_csv(id_path, sep="\t", header=None).values.flatten()
    
    all_index["sequence"] = {"index": sequence_index, "ids": uniprot_ids}
    
    # Load protein structure index
    index_path = f"{index_dir}/structure.index"
    structure_index = faiss.read_index(index_path)
    
    id_path = f"{index_dir}/structure_ids.tsv"
    uniprot_ids = pd.read_csv(id_path, sep="\t", header=None).values.flatten()
    
    all_index["structure"] = {"index": structure_index, "ids": uniprot_ids}
    
    # Load text index
    all_index["text"] = {}
    text_dir = f"{index_dir}/text"
    
    # Remove "Taxonomic lineage" from sequence_level. This is a special case which we don't need to index.
    valid_subsections = set()
    sequence_level.add("Global")
    for subsection in sequence_level:
        index_path = f"{text_dir}/{subsection.replace(' ', '_')}.index"
        if not os.path.exists(index_path):
            continue
            
        text_index = faiss.read_index(index_path)
        
        id_path = f"{text_dir}/{subsection.replace(' ', '_')}_ids.tsv"
        text_ids = pd.read_csv(id_path, sep="\t", header=None).values.flatten()
        
        all_index["text"][subsection] = {"index": text_index, "ids": text_ids}
        valid_subsections.add(subsection)

    return all_index, valid_subsections


device = "cuda"

print("Loading model...")
model = load_model()
model.to(device)

print("Loading index...")
all_index, valid_subsections = load_index()
print("Done...")
# model = None
# all_index, valid_subsections = {"text": {}}, {}