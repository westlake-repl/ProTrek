import os
import yaml

from easydict import EasyDict
from utils.constants import sequence_level
from utils.file_reader import FileReader
from utils.faiss_index import FaissIndex
from tqdm import tqdm
from typing import List


ROOT_DIR = __file__.rsplit("/", 5)[0]
config_path = f"{ROOT_DIR}/demo/config.yaml"


def load_index(config):
    all_index = {}

    # Load protein sequence index
    all_index["sequence"] = {}
    for db in tqdm(config.sequence_index_dir, desc="Loading sequence index..."):
        db_name = db["name"]
        index_dir = f"{ROOT_DIR}/{db['index_dir']}"

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
                if not os.path.exists(index_path):
                    continue

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
        index_dir = f"{ROOT_DIR}/{db['index_dir']}"

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
        index_dir = f"{ROOT_DIR}/{db['index_dir']}"
        all_index["text"][db_name] = {}
        text_dir = f"{index_dir}/subsections"

        # Remove "Taxonomic lineage" from sequence_level. This is a special case which we don't need to index.
        valid_subsections[db_name] = set()
        sequence_level.add("Global")
        sequence_level.add("Enzyme commission number")
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


def init_index(sequence_db: List[str] = None, structure_db: List[str] = None, text_db: List[str] = None):
    """
    Args:
        sequence_db: Specify which sequence databases to use. If None, follow the config file.
        structure_db: Specify which structure databases to use. If None, follow the config file.
        text_db: Specify which text databases to use. If None, follow the config file.
    """

    # Load the config file
    with open(config_path, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r)).retrieval
    
    # Override the config file with the specified databases
    if sequence_db is not None:
        sequence_index_dir = []
        for db in config.sequence_index_dir:
            if db["name"] in sequence_db:
                sequence_index_dir.append(db)
        
        config.sequence_index_dir = sequence_index_dir
    
    if structure_db is not None:
        structure_index_dir = []
        for db in config.structure_index_dir:
            if db["name"] in structure_db:
                structure_index_dir.append(db)
        
        config.structure_index_dir = structure_index_dir
    
    if text_db is not None:
        text_index_dir = []
        for db in config.text_index_dir:
            if db["name"] in text_db:
                text_index_dir.append(db)
        
        config.text_index_dir = text_index_dir
    
    all_index, valid_subsections = load_index(config)
    return all_index, valid_subsections
