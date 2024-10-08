import os
import sys

sys.path += ["."]

import re
import argparse
import torch
import numpy as np
import json

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from Bio import SeqIO
from utils.mpr import MultipleProcessRunnerSimplifier


def main(args):
    if os.path.isdir(args.input):
        print("Input is a directory, will predict all fasta files in the directory.")

    else:
        print("Input is a fasta file.")

    n_process = 1
    use_gpu = False
    if "cpu" in args.device.lower():
        print("CUDA is not specified. Will use CPU.")

    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        gpu_num = len(args.device.split(","))
        if gpu_num > 1:
            print("Multi-GPU is expected, will use multiple GPUs.")
            n_process = gpu_num

    print("Loading model from {}...".format(args.model_path))
    tokenizer, model = load_model(args.model_path)
    model.eval()
    print("Model loaded.")

    fasta_list = []
    save_path_list = []
    if os.path.isdir(args.input):
        for f in os.listdir(args.input):
            name, _ = os.path.splitext(f)

            fasta_list.append(os.path.join(args.input, f))
            save_path_list.append(os.path.join(args.save_path, name + ".pdb"))

    else:
        fasta_list.append(args.input)
        save_path_list.append(args.save_path)

    overwrite = args.overwrite

    def do(process_id, idx, input, writer):
        try:
            if use_gpu:
                device = torch.device(f"cuda:{process_id}")
                if model.device != device:
                    model.to(device)

            fasta, save_path = input
            if os.path.exists(save_path) and not overwrite:
                return

            for seq_record in SeqIO.parse(fasta, "fasta"):
                seq = str(seq_record.seq)
                predict(seq, tokenizer, model, save_path)

        except Exception as e:
            print(e)
            print("Error on {}".format(fasta))

    data = [i for i in zip(fasta_list, save_path_list)]

    print("Predicting...")
    mprs = MultipleProcessRunnerSimplifier(data, do, save_path=None, n_process=n_process,
                                           total_only=True, start_method="fork", split_strategy="queue")
    mprs.run()
    print("Done.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', help="Fasta file that contains protein sequences to build the database",
                        type=str, required=True)

    parser.add_argument('--save_dir', help="Save the database to the directory", type=str, required=True)

    parser.add_argument('--device', help="Running inference on specific device. If "
                                         "multi-gpu is expected, set gpu number seperated by comma, "
                                         "e.g. '0,1,2,3'. default: cpu", type=str, default="cpu")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
