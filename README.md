# ProTrek
<a href="https://huggingface.co/spaces/westlake-repl/Demo_ProTrek_650M_UniRef50"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-red?label=Demo" style="max-width: 100%;"></a>

<details open><summary><b>Table of contents</b></summary>

- [Overview](#Overview)
- [Environment installation](#Environment-installation)
- [Download model weights](#Download-model-weights)
- [Download Foldseek binary file](#Download-Foldseek-binary-file)
- [Obtain embeddings and calculate similarity score](#Obtain-embeddings-and-calculate-similarity-score)
- [Deploy your demo locally](#Deploy-your-demo-locally)
</details>

## Overview
ProTrek is a multimodal model that integrates protein sequence, protein structure, and text information for better 
protein understanding. It adopts contrastive learning to learn the representations of protein sequence and structure. 
During the pre-training phase, we calculate the InfoNCE loss for each two modalities as [CLIP](https://arxiv.org/abs/2103.00020)
does.

<img src="figure/img.png" style="zoom:33%;" />

## Environment installation
### Create a virtual environment
```
conda create -n protrek python=3.10 --yes
conda activate protrek
```
### Clone the repo and install packages
```
bash environment.sh  
```

## Download model weights
ProTrek provides pre-trained models with different sizes (35M and 650M), as shown below. For each pre-trained model, 
Please download all files and put them in the `weights` directory, e.g. `weights/ProTrek_35M_UniRef50/...`.


| **Name**                                                     | **Size （protein sequence encoder）** | **Size （protein structure encoder）** | **Size （text encoder）** | Dataset               |
| ------------------------------------------------------------ | ------------------------------------- | -------------------------------------- | ------------------------- | --------------------- |
| [ProTrek_35M_UniRef50](https://huggingface.co/westlake-repl/ProTrek_35M_UniRef50) | 35M parameters                        | 35M parameters                         | 110M parameters           | Swiss-Prot + UniRef50 |
| [ProTrek_650M_UniRef50](https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50) | 650M parameters                       | 150M parameters                        | 110M parameters           | Swiss-Prot + UniRef50 |

We provide an example to download the pre-trained model weights.
```
huggingface-cli download westlake-repl/ProTrek_650M_UniRef50 \
                         --repo-type model \
                         --local-dir weights/ProTrek_650M_UniRef50
```
> Note: if you cannot access the huggingface website, you can try to connect to the mirror site through "export 
> HF_ENDPOINT=https://hf-mirror.com"

## Download Foldseek binary file
To run examples correctly and deploy your demo locally, please at first download the [Foldseek](https://github.com/steineggerlab/foldseek) 
binary file from [here](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing) and place 
it in the `bin` folder. Then add the execute permission to the binary file.
```
chmod +x bin/foldseek
```

## Obtain embeddings and calculate similarity score
Below is an example of how to obtain embeddings and calculate similarity score using the pre-trained ProTrek model.
```
import torch

from model.ProtTrek.protrek_trimodal_model import ProTrekTrimodalModel
from utils.foldseek_util import get_struc_seq

# Load model
config = {
    "protein_config": "weights/ProTrek_650M_UniRef50/esm2_t33_650M_UR50D",
    "text_config": "weights/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "weights/ProTrek_650M_UniRef50/foldseek_t30_150M",
    "load_protein_pretrained": False,
    "load_text_pretrained": False,
    "from_checkpoint": "weights/ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt"
}

device = "cuda"
model = ProTrekTrimodalModel(**config).eval().to(device)

# Load protein and text
pdb_path = "example/8ac8.cif"
seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"])["A"]
aa_seq = seqs[0]
foldseek_seq = seqs[1].lower()
text = "Replication initiator in the monomeric form, and autogenous repressor in the dimeric form."

with torch.no_grad():
    # Obtain protein sequence embedding
    seq_embedding = model.get_protein_repr([aa_seq])
    print("Protein sequence embedding shape:", seq_embedding.shape)
    
    # Obtain protein structure embedding
    struc_embedding = model.get_structure_repr([foldseek_seq])
    print("Protein structure embedding shape:", struc_embedding.shape)
    
    # Obtain text embedding
    text_embedding = model.get_text_repr([text])
    print("Text embedding shape:", text_embedding.shape)
    
    # Calculate similarity score between protein sequence and structure
    seq_struc_score = seq_embedding @ struc_embedding.T / model.temperature
    print("Similarity score between protein sequence and structure:", seq_struc_score.item())

    # Calculate similarity score between protein sequence and text
    seq_text_score = seq_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein sequence and text:", seq_text_score.item())
    
    # Calculate similarity score between protein structure and text
    struc_text_score = struc_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein structure and text:", struc_text_score.item())
   

"""
Protein sequence embedding shape: torch.Size([1, 1024])
Protein structure embedding shape: torch.Size([1, 1024])
Text embedding shape: torch.Size([1, 1024])
Similarity score between protein sequence and structure: 28.506675720214844
Similarity score between protein sequence and text: 17.842409133911133
Similarity score between protein structure and text: 11.866174697875977
"""
```

## Deploy your demo locally
We provide an [online demo](https://huggingface.co/spaces/westlake-repl/Demo_ProTrek_650M_UniRef50) for ProTrek. For users who want to deploy the demo locally, please follow the steps below.

### Step 1: Download the Foldseek binary file
Please follow the instructions in the [Download Foldseek binary file](#Download-Foldseek-binary-file) section.

### Step 2: Download the pre-trained model weights
Currently we support the deployment of [ProTrek_650M_UniRef50](https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50).
Please download all files and put them in the `weights` directory, e.g. `weights/ProTrek_650M_UniRef50/...`. The example
code is in the [Download model weights](#Download-model-weights) section.

### Step 3: Download pre-computed faiss index
We provide pre-computed protein embeddings and text embeddings using [ProTrek_650M_UniRef50](https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50),
and build faiss index for fast similarity search. Please download the pre-computed faiss index from [here](https://huggingface.co/datasets/westlake-repl/faiss_index_ProTrek_650M_UniRef50/tree/main)
and put it in the `weights/faiss_index` directory, e.g. `weights/faiss_index/faiss_index_ProTrek_650M_UniRef50/...`. We
provide an example to download the pre-computed faiss index.
```
huggingface-cli download westlake-repl/faiss_index_ProTrek_650M_UniRef50 \
                         --repo-type dataset \
                         --local-dir weights/faiss_index/faiss_index_ProTrek_650M_UniRef50
```

### Step 4: Run the demo
After all data and files are prepared, you can run the demo by executing the following command.
```
python demo/run.py 
```