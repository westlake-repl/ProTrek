import os
import yaml

from easydict import EasyDict
from utils.constants import sequence_level
from utils.file_reader import FileReader
from utils.faiss_index import FaissIndex
from tqdm import tqdm


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


# Load the config file
base_dir = os.path.dirname(__file__)
config_path = f"{base_dir}/config.yaml"
with open(config_path, 'r', encoding='utf-8') as r:
    config = EasyDict(yaml.safe_load(r))

all_index, valid_subsections = load_index()
print("Done...")
