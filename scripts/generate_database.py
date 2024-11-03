import os
import sys

sys.path += ["."]

import argparse
import torch
import faiss
import glob
import numpy as np

from Bio import SeqIO
from utils.mpr import MultipleProcessRunnerSimplifier
from tqdm import tqdm
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation."
    n_process = torch.cuda.device_count()

    if args.device != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        gpu_num = len(args.device.split(","))
        n_process = gpu_num

        print(f"Specified devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    ##########################################
    #         Load protein sequences         #
    ##########################################
    os.makedirs(args.save_dir, exist_ok=True)
    id_path = os.path.join(args.save_dir, "ids.tsv")
    cnt = 0
    items = []
    warning_flag = False
    with open(id_path, "w") as w:
        for record in tqdm(SeqIO.parse(args.fasta, "fasta")):
            id = record.id
            seq = str(record.seq)
            if len(seq) > 2048:
                if not warning_flag:
                    print(f"Warning: Sequence greater than 2048 will be skipped.")
                    warning_flag = True
                continue

            w.write(f"{id}\t{seq}\t{len(seq)}\n")
            items.append((cnt, seq))
            cnt += 1

    assert cnt < 10000001, "The number of sequences should not be greater than 10000000."

    ##########################################
    #       Compute protein embeddings       #
    ##########################################
    root_dir = os.path.abspath(__file__).rsplit("/", 2)[0]

    # Load the model
    model_config = {
        "protein_config": glob.glob(f"{root_dir}/weights/ProTrek_650M_UniRef50/esm2_*")[0],
        "text_config": f"{root_dir}/weights/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": glob.glob(f"{root_dir}/weights/ProTrek_650M_UniRef50/foldseek_*")[0],
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": glob.glob(f"{root_dir}/weights/ProTrek_650M_UniRef50/*.pt")[0]
    }

    model = ProTrekTrimodalModel(**model_config)
    model.eval()

    # Create empty embeddings
    npy_path = os.path.join(args.save_dir, f"embeddings_{cnt}.npy")
    if os.path.exists(npy_path):
        embeddings = np.memmap(npy_path, dtype=np.float32, mode="r+", shape=(cnt, 1024))
    else:
        embeddings = np.memmap(npy_path, dtype=np.float32, mode="write", shape=(cnt, 1024))

    # Fill embeddings
    def do(process_id, idx, item, writer):
        if model.device == torch.device("cpu"):
            device = f"cuda:{process_id % n_process}"
            model.to(device)

        i, seq = item
        with torch.no_grad():
            # Skip pre-computed embeddings
            if embeddings[i].sum() != 0:
                return

            seq_repr = model.get_protein_repr([seq])
            embeddings[i] = seq_repr.cpu().numpy()

    mprs = MultipleProcessRunnerSimplifier(items, do, n_process=n_process*2, split_strategy="queue", log_step=1000)
    mprs.run()

    ##########################################
    #           Build Faiss index            #
    ##########################################
    if len(embeddings) < 1000000:
        # Use brute-force search for small dataset
        index = faiss.IndexFlatIP(1024)
    else:
        # Use IVF for large dataset
        n_cluster = min(len(embeddings) // 39, 65536)
        quantizer = faiss.IndexFlatIP(1024)
        index = faiss.IndexIVFFlat(quantizer, 1024, n_cluster, faiss.METRIC_INNER_PRODUCT)
        print(n_cluster)
    
    # Train the index if it requires training
    if not index.is_trained:
        print("Building index...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.train(embeddings)
        index = faiss.index_gpu_to_cpu(index)
    
    for i in tqdm(range(0, len(embeddings), 100000), desc="Adding embeddings to index..."):
        e = embeddings[i:i+100000]
        index.add(e)
   
    index_path = os.path.join(args.save_dir, "sequence.index")
    faiss.write_index(index, index_path)
    print("Done.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', help="Fasta file that contains protein sequences to build the database",
                        type=str, required=True)

    parser.add_argument('--save_dir', help="Save the database to the directory", type=str, required=True)

    parser.add_argument('--device', help="Running inference on specific device. If "
                                         "multiple GPUs are expected, set GPU number seperated by comma, "
                                         "e.g. '0,1,2,3'. default: all available GPUs", type=str, default="")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
