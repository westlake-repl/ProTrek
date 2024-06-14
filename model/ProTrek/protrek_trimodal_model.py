import torch
import torch.distributed as dist
import torchmetrics
import json
import math
import numpy as np
import os
import copy
import faiss
import time
import pandas as pd
import random

from tqdm import tqdm
from .protein_encoder import ProteinEncoder
from .structure_encoder import StructureEncoder
from .text_encoder import TextEncoder
from ..abstract_model import AbstractModel
from ..model_interface import register_model
from utils.mpr import MultipleProcessRunnerSimplifier
from torch.nn.functional import normalize, cross_entropy
from utils.constants import residue_level, sequence_level
from sklearn.metrics import roc_auc_score


def multilabel_cross_entropy(logits, labels):
    """
    Compute cross entropy loss for multilabel classification。 See "https://arxiv.org/pdf/2208.02955.pdf"
    Args:
        logits: [num_samples, num_classes]
        labels: [num_samples, num_classes]
    """

    loss = 0
    for pred, label in zip(logits, labels):
        pos_logits = pred[label == 1]
        neg_logits = pred[label == 0]

        diff = neg_logits.unsqueeze(-1) - pos_logits
        loss += torch.log(1 + torch.exp(diff).sum())

    return loss / len(logits)

    # pred = (1 - 2 * labels) * logits
    # pred_neg = pred - labels * 1e12
    # pred_pos = pred - (1 - labels) * 1e12
    #
    # zeros = torch.zeros_like(logits[..., :1], dtype=logits.dtype)
    # pred_neg = torch.cat([pred_neg, zeros], dim=-1)
    # pred_pos = torch.cat([pred_pos, zeros], dim=-1)
    #
    # neg_loss = torch.logsumexp(pred_neg, dim=-1)
    # pos_loss = torch.logsumexp(pred_pos, dim=-1)
    #
    # return (neg_loss + pos_loss).mean()


@register_model
class ProTrekTrimodalModel(AbstractModel):
    def __init__(self,
                 protein_config: str,
                 text_config: str,
                 structure_config: str = None,
                 repr_dim: int = 1024,
                 temperature: float = 0.07,
                 load_protein_pretrained: bool = True,
                 load_text_pretrained: bool = True,
                 use_mlm_loss: bool = False,
                 use_zlpr_loss: bool = False,
                 use_saprot: bool = False,
                 gradient_checkpointing: bool = False,
                 **kwargs):
        """
        Args:
            protein_config: Path to the config file for protein sequence encoder
            
            text_config: Path to the config file for text encoder
            
            structure_config: Path to the config file for structure encoder
            
            repr_dim: Output dimension of the protein and text representation
            
            temperature: Temperature for softmax

            load_protein_pretrained: Whether to load pretrained weights for protein encoder

            load_text_pretrained: Whether to load pretrained weights for text encoder

            use_mlm_loss: Whether to use masked language modeling loss
            
            use_zlpr_loss: Whether to use zlpr loss. See "https://arxiv.org/pdf/2208.02955.pdf"

            use_saprot: Whether to use SaProt as protein encoder
            
            gradient_checkpointing: Whether to use gradient checkpointing for protein encoder
        """
        self.protein_config = protein_config
        self.structure_config = structure_config
        self.text_config = text_config
        self.repr_dim = repr_dim
        self.temperature = temperature
        self.load_protein_pretrained = load_protein_pretrained
        self.load_text_pretrained = load_text_pretrained
        self.use_mlm_loss = use_mlm_loss
        self.use_zlpr_loss = use_zlpr_loss
        self.use_saprot = use_saprot
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(**kwargs)
    
    def initialize_metrics(self, stage: str) -> dict:
        return_dict = {
            f"{stage}_protein_text_acc": torchmetrics.Accuracy(),
            f"{stage}_text_protein_acc": torchmetrics.Accuracy(),
        }
        
        if self.use_mlm_loss:
            return_dict[f"{stage}_protein_mask_acc"] = torchmetrics.Accuracy(ignore_index=-1)
            if self.structure_config is not None:
                return_dict[f"{stage}_structure_mask_acc"] = torchmetrics.Accuracy(ignore_index=-1)
        
        if self.structure_config is not None:
            return_dict[f"{stage}_structure_protein_acc"] = torchmetrics.Accuracy()
            return_dict[f"{stage}_structure_text_acc"] = torchmetrics.Accuracy()
            return_dict[f"{stage}_text_structure_acc"] = torchmetrics.Accuracy()
            return_dict[f"{stage}_protein_structure_acc"] = torchmetrics.Accuracy()

        return return_dict

    def initialize_model(self):
        # Initialize encoders
        self.protein_encoder = ProteinEncoder(self.protein_config,
                                              self.repr_dim,
                                              self.load_protein_pretrained,
                                              self.gradient_checkpointing)
        
        self.text_encoder = TextEncoder(self.text_config,
                                        self.repr_dim,
                                        self.load_text_pretrained,
                                        self.gradient_checkpointing)
        
        # Learnable temperature
        self.temperature = torch.nn.Parameter(torch.tensor(self.temperature))
        
        # self.model is used for saving and loading
        self.model = torch.nn.ParameterList([self.temperature,
                                             self.protein_encoder,
                                             self.text_encoder])
        
        # If the structure encoder is specified
        if self.structure_config is not None:
            self.structure_encoder = StructureEncoder(self.structure_config, self.repr_dim)
            self.model.append(self.structure_encoder)
    
    def get_text_repr(self, texts: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        return self.text_encoder.get_repr(texts, batch_size, verbose)
    
    def get_structure_repr(self, proteins: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        return self.structure_encoder.get_repr(proteins, batch_size, verbose)
    
    def get_protein_repr(self, proteins: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        return self.protein_encoder.get_repr(proteins, batch_size, verbose)
    
    def forward(self, protein_inputs: dict, text_inputs: dict, structure_inputs: dict = None):
        """
        Args:
            protein_inputs: A dictionary for protein encoder
            structure_inputs: A dictionary for structure encoder
            text_inputs   : A dictionary for text encoder
        """
        protein_repr, protein_mask_logits = self.protein_encoder(protein_inputs, self.use_mlm_loss)
        text_repr = self.text_encoder(text_inputs)
        
        outputs = [text_repr, protein_repr, protein_mask_logits]
        
        if self.structure_config is not None:
            structure_repr, structure_mask_logits = self.structure_encoder(structure_inputs, self.use_mlm_loss)
            outputs += [structure_repr, structure_mask_logits]
        
        return outputs
    
    def loss_func(self, stage: str, outputs, labels):
        if self.structure_config is not None:
            text_repr, protein_repr, protein_mask_logits, structure_repr, structure_mask_logits = outputs
        else:
            text_repr, protein_repr, protein_mask_logits = outputs

        device = text_repr.device

        text_repr = normalize(text_repr, dim=-1)
        protein_repr = normalize(protein_repr, dim=-1)
        
        # Gather representations from all GPUs
        all_protein_repr = self.all_gather(protein_repr).view(-1, protein_repr.shape[-1]).detach()
        all_text_repr = self.all_gather(text_repr).view(-1, text_repr.shape[-1]).detach()

        if self.structure_config is not None:
            structure_repr = normalize(structure_repr, dim=-1)
            all_structure_repr = self.all_gather(structure_repr).view(-1, structure_repr.shape[-1]).detach()

        # text_idx = labels["text_idx"]
        # text_candidates = labels["text_candidates"]
        #
        # # Gather all text ids
        # text_inds = self.all_gather(text_idx).flatten()
        # # Create text classification labels
        # text_labels = torch.zeros(len(text_candidates), len(text_inds), dtype=int).to(device)
        # for i, candidate in enumerate(text_candidates):
        #     for j, idx in enumerate(text_inds):
        #         if idx.item() in candidate:
        #             text_labels[i, j] = 1
        #
        # # Gather text labels from all GPUs
        # text_labels = self.all_gather(text_labels).view(-1, text_labels.shape[-1])
        #
        # # Protein classification labels are the transpose of text labels
        # protein_labels = text_labels.T

        # Batch size
        rank = dist.get_rank()
        bs = text_repr.shape[0]
    
        # Get current labels
        # protein_labels = protein_labels[rank * bs: rank * bs + bs]
        # text_labels = text_labels[rank * bs: rank * bs + bs]

        # Create classification labels between structure and sequence
        bs_labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(device)

        if self.structure_config is not None:
            pairs = {
                "protein": ["structure", "text"],
                "structure": ["protein", "text"],
                "text": ["protein", "structure"]
            }
        else:
            pairs = {
                "protein": ["text"],
                "text": ["protein"]
            }

        loss_list = []
        for k, values in pairs.items():
            for v in values:
                # Only calculate the similarity for the current batch
                sim = torch.matmul(eval(f"{k}_repr"), eval(f"all_{v}_repr").T).div(self.temperature)
                
                # if k == "text":
                #     if self.use_zlpr_loss:
                #         loss = multilabel_cross_entropy(sim, protein_labels)
                #     else:
                #         loss = cross_entropy(sim, bs_labels)
                #
                #     pred = []
                #     for s, l in zip(sim, protein_labels):
                #         n_label = l.sum()
                #         topk = torch.topk(s, k=n_label).indices
                #         if l[topk].sum() == n_label:
                #             pred.append(1)
                #         else:
                #             pred.append(0)
                #
                #     pred = torch.tensor(pred).to(device)
                #     label = torch.ones_like(pred)
                #     self.metrics[stage][f"{stage}_{k}_{v}_acc"].update(pred.detach(), label)
                #     # if v == "protein":
                #     #     acc = self.metrics[stage][f"{stage}_{k}_{v}_acc"].compute()
                #     #     print(f"{stage}_{k}_{v}_acc: {acc:.4f}")
                #
                # elif v == "text":
                #     if self.use_zlpr_loss:
                #         loss = multilabel_cross_entropy(sim, text_labels)
                #     else:
                #         loss = cross_entropy(sim, bs_labels)
                #
                #     pred = []
                #     for s, l in zip(sim, text_labels):
                #         n_label = l.sum()
                #         topk = torch.topk(s, k=n_label).indices
                #         if l[topk].sum() == n_label:
                #             pred.append(1)
                #         else:
                #             pred.append(0)
                #
                #     pred = torch.tensor(pred).to(device)
                #     label = torch.ones_like(pred)
                #     # if k == "protein":
                #     #     acc = pred.sum() / len(pred)
                #     #     print(f"{stage}_{k}_{v}_acc: {acc:.4f}")
                #     self.metrics[stage][f"{stage}_{k}_{v}_acc"].update(pred.detach(), label)
                #
                # else:
                #     loss = cross_entropy(sim, bs_labels)
                #     self.metrics[stage][f"{stage}_{k}_{v}_acc"].update(sim.detach(), bs_labels)

                loss = cross_entropy(sim, bs_labels)
                self.metrics[stage][f"{stage}_{k}_{v}_acc"].update(sim.detach(), bs_labels)
                loss_list.append(loss)

        # Masked language modeling loss
        if self.use_mlm_loss:
            k_label = [("protein", labels["seq_labels"])]
            if self.structure_config is not None:
                k_label.append(("structure", labels["struc_labels"]))

            for k, label in k_label:
                logits = eval(f"{k}_mask_logits")
                # merge the first and second dimension of logits
                logits = logits.view(-1, logits.shape[-1])
                label = label.flatten().to(device)
                mlm_loss = cross_entropy(logits, label, ignore_index=-1)
                loss_list.append(mlm_loss)
                self.metrics[stage][f"{stage}_{k}_mask_acc"].update(logits.detach(), label)

        loss = sum(loss_list) / len(loss_list)
        
        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)
            
            # Reset train metrics
            self.reset_metrics("train")
        
        return loss
    
    def padded_gather(self, tensor: torch.Tensor):
        """
        Gather tensors from all GPUs, allowing different shapes at the batch dimension.
        """
        
        # Get the size of the tensor
        size = tensor.shape[0]
        all_sizes = self.all_gather(torch.tensor(size, device=tensor.device))
        max_size = max(all_sizes)
        
        # Pad the tensor
        if size != max_size:
            tmp = torch.zeros(max_size, tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
            tmp[:size] = tensor
            tensor = tmp
        
        padded_tensor = self.all_gather(tensor).view(-1, tensor.shape[-1])
        tensor = padded_tensor[:sum(all_sizes)]
        
        return tensor

    def _get_protein_indices(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if self.use_saprot:
            proteins = []
            for sub_dict in self.uniprot2label.values():
                aa_seq = sub_dict["seq"]
                foldseek_seq = sub_dict["foldseek"]
                assert len(aa_seq) == len(foldseek_seq)
                seq = "".join([a + b for a, b in zip(aa_seq, foldseek_seq)])
                proteins.append(seq)

        else:
            proteins = [sub_dict["seq"] for sub_dict in self.uniprot2label.values()]

        span = math.ceil(len(proteins) / world_size)
        sub_proteins = proteins[rank * span: (rank + 1) * span]
        
        # Display the progress bar on the rank 0 process
        verbose = self.trainer.local_rank == 0
        # Get protein representations
        sub_protein_repr = self.protein_encoder.get_repr(sub_proteins, batch_size=1, verbose=verbose)
        protein_repr = self.padded_gather(sub_protein_repr)
        
        # Construct faiss index
        d = protein_repr.shape[-1]
        protein_indices = faiss.IndexFlatIP(d)
        protein_indices.add(protein_repr.cpu().numpy())
        return protein_indices
    
    def _get_structure_indices(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        proteins = [sub_dict["foldseek"] for sub_dict in self.uniprot2label.values()]
        span = math.ceil(len(proteins) / world_size)
        sub_proteins = proteins[rank * span: (rank + 1) * span]
        
        # Display the progress bar on the rank 0 process
        verbose = self.trainer.local_rank == 0
        # Get protein representations
        sub_protein_repr = self.structure_encoder.get_repr(sub_proteins, batch_size=1, verbose=verbose)
        protein_repr = self.padded_gather(sub_protein_repr)
        
        # Construct faiss index
        d = protein_repr.shape[-1]
        structure_indices = faiss.IndexFlatIP(d)
        structure_indices.add(protein_repr.cpu().numpy())
        return structure_indices
    
    def _get_text_indices(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Display the progress bar on the rank 0 process
        verbose = self.trainer.local_rank == 0
        if verbose:
            iterator = tqdm(self.label2text.keys(), desc="Get text representations")
        else:
            iterator = self.label2text.keys()
        
        text_embeddings = {}
        for subsection in iterator:
            if subsection == "Total":
                continue
                
            texts = []
            for text_list in self.label2text[subsection].values():
                # Only use the first text for efficiency
                texts.append(text_list[0:1])
            
            span = math.ceil(len(texts) / world_size)
            texts = texts[rank * span: (rank + 1) * span]
            embeddings = []
            for text_list in texts:
                text_repr = self.text_encoder.get_repr(text_list)
                mean_repr = text_repr.mean(dim=0, keepdim=True)
                norm_repr = torch.nn.functional.normalize(mean_repr, dim=-1)
                embeddings.append(norm_repr)
            
            if len(embeddings) > 0:
                embeddings = torch.cat(embeddings, dim=0)
            else:
                embeddings = torch.zeros(0, self.repr_dim, dtype=self.dtype, device=self.device)
            
            text_repr = self.padded_gather(embeddings)
            text_embeddings[subsection] = text_repr
        
        # Aggregate text embeddings for global retrieval
        total_embeddings = []
        for idx in self.label2text["Total"].values():
            subsection, i = idx.split("|")
            total_embeddings.append(text_embeddings[subsection][int(i)])
        
        text_embeddings["Total"] = torch.stack(total_embeddings)
        
        # Construct faiss index
        text_indices = {}
        for subsection, text_repr in text_embeddings.items():
            d = text_repr.shape[-1]
            text_indices[subsection] = faiss.IndexFlatIP(d)
            text_indices[subsection].add(text_repr.cpu().numpy())
        
        return text_indices
    
    def _protein2text(self, modality: str, protein_indices, text_indices: dict):
        def do(process_id, idx, row, writer):
            subsection, uniprot_id, prob_idx, label = row
            
            # Retrieve ranking results
            p_embedding = protein_indices.reconstruct(prob_idx).reshape(1, -1)
            text_inds = text_indices[subsection]
            sim_scores, rank_inds = text_inds.search(p_embedding, text_inds.ntotal)
            sim_scores, rank_inds = sim_scores[0], rank_inds[0]
            
            # Calculate Average Precision(AP)
            ranks = []
            label = set(label)
            for i, rk in enumerate(rank_inds):
                # Find the rank of this label in all labels
                if rk in label:
                    ranks.append(i + 1)
            
            ranks = np.array(ranks)
            ap = np.mean([(i + 1) / rank for i, rank in enumerate(ranks)])
            
            # Calculate Mean Reciprocal Rank(MRR)
            best_rank = ranks[0]
            mrr = 1 / best_rank
            
            # Calculate the AUC
            true_labels = np.zeros_like(sim_scores)
            true_labels[ranks - 1] = 1
            if true_labels.sum() == 0 or true_labels.sum() == true_labels.shape[0]:
                auc = 0
            else:
                auc = roc_auc_score(true_labels, sim_scores)
            
            output = json.dumps([ap, mrr, auc])
            writer.write(output + "\n")
            
        inputs = []
        swissprot_subsections = set()
        for subsection in text_indices.keys():
            for i, (uniprot_id, labels) in enumerate(self.uniprot2label.items()):
                if uniprot_id in self.swissprot_ids:
                    if subsection in labels:
                        swissprot_subsections.add(subsection)
                        label = labels[subsection]
                        inputs.append((subsection, uniprot_id, i, label))
                
        # Randomly shuffle the inputs
        random.seed(20000812)
        random.shuffle(inputs)
        
        # Split inputs into chunks for parallel processing
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        span = math.ceil(len(inputs) / world_size)
        sub_inputs = inputs[rank * span: (rank + 1) * span]

        # Display the progress bar on the rank 0 process
        verbose = self.trainer.local_rank == 0
        if verbose:
            print("Evaluating on each subsection...")
        tmp_path = f"/sujin/PycharmProjects/Pretraining/{time.time()}_{rank}.tsv"
        mpr = MultipleProcessRunnerSimplifier(sub_inputs, do, save_path=tmp_path, n_process=8, verbose=verbose,
                                              return_results=True)
        outputs = mpr.run()
        os.remove(tmp_path)
        
        # Aggregate results
        tensor_outputs = []
        for output in outputs:
            ap, mrr, auc = json.loads(output)
            tensor_outputs.append([float(ap), float(mrr), float(auc)])

        tensor_outputs = torch.tensor(tensor_outputs, dtype=torch.float32, device=self.device)
        tensor_outputs = self.padded_gather(tensor_outputs)
        
        # Record results
        avg_results = {}
        for subsection in swissprot_subsections:
            avg_results[subsection] = {"map": [],
                                       "mrr": [],
                                       "auc": []}
        
        for input, output in zip(inputs, tensor_outputs):
            ap, mrr, auc = output
            subsection, _, _, _ = input
        
            avg_results[subsection]["map"].append(ap.cpu().item())
            avg_results[subsection]["mrr"].append(mrr.cpu().item())
            avg_results[subsection]["auc"].append(auc.cpu().item())
        
        results = {
            f"{modality}2Text_Total_mrr": np.mean(avg_results["Total"]["mrr"]),
            f"{modality}2Text_Total_map": np.mean(avg_results["Total"]["map"]),
            f"{modality}2Text_Total_auc": np.mean(avg_results["Total"]["auc"]),
        }
        
        # Average the precision and recall for each level
        for level, labels in [("residue-level", residue_level),
                              ("sequence-level", sequence_level),
                              ("all", residue_level | sequence_level)]:
            
            mrrs = []
            maps = []
            aucs = []
            for subsection in labels:
                if subsection in avg_results:
                    mrrs.append(np.mean(avg_results[subsection]["mrr"]))
                    maps.append(np.mean(avg_results[subsection]["map"]))
                    aucs.append(np.mean(avg_results[subsection]["auc"]))
            
            results[f"{modality}2Text_{level}_mrr"] = np.mean(mrrs)
            results[f"{modality}2Text_{level}_map"] = np.mean(maps)
            results[f"{modality}2Text_{level}_auc"] = np.mean(aucs)

        return results
    
    def _text2protein(self, modality: str, protein_indices, text_indices: dict):
        def do(process_id, idx, row, writer):
            subsection, text_id, label = row

            # Retrieve ranking results
            t_embedding = text_indices[subsection].reconstruct(text_id).reshape(1, -1)
            sim_scores, rank_inds = protein_indices.search(t_embedding, protein_indices.ntotal)
            sim_scores, rank_inds = sim_scores[0], rank_inds[0]

            # Calculate Average Precision(AP)
            ranks = []
            label = set(label)
            for i, rk in enumerate(rank_inds):
                # Find the rank of this label in all labels
                if rk in label:
                    ranks.append(i + 1)

            ranks = np.array(ranks)
            ap = np.mean([(i + 1) / rank for i, rank in enumerate(ranks)])

            # Calculate Mean Reciprocal Rank(MRR)
            best_rank = ranks[0]
            mrr = 1 / best_rank

            # Calculate the AUC
            true_labels = np.zeros_like(sim_scores)
            true_labels[ranks - 1] = 1
            if true_labels.sum() == 0 or true_labels.sum() == true_labels.shape[0]:
                auc = 0
            else:
                auc = roc_auc_score(true_labels, sim_scores)

            output = json.dumps([ap, mrr, auc])
            writer.write(output + "\n")

        text2label = {}
        swissprot_subsections = set()
        for i, (uniprot_id, subsections) in enumerate(self.uniprot2label.items()):
            # Only evaluate the texts in Swiss-Prot
            if uniprot_id not in self.swissprot_ids:
                continue
                
            for subsection, text_ids in subsections.items():
                if subsection == "seq" or subsection == "foldseek":
                    continue

                swissprot_subsections.add(subsection)
                if subsection not in text2label:
                    text2label[subsection] = {}
                
                for text_id in text_ids:
                    text2label[subsection][text_id] = text2label[subsection].get(text_id, []) + [i]

        inputs = []
        for subsection in swissprot_subsections:
            for i, (text_id, label) in enumerate(text2label[subsection].items()):
                inputs.append((subsection, text_id, label))

        # Randomly shuffle the inputs
        random.seed(20000812)
        random.shuffle(inputs)

        # Split inputs into chunks for parallel processing
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        span = math.ceil(len(inputs) / world_size)
        sub_inputs = inputs[rank * span: (rank + 1) * span]

        # Display the progress bar on the rank 0 process
        verbose = self.trainer.local_rank == 0
        if verbose:
            print("Evaluating on each text...")
            
        # Add time stamp to the temporary file name to avoid conflicts
        tmp_path = f"/sujin/PycharmProjects/Pretraining/{time.time()}_{rank}.tsv"
        mpr = MultipleProcessRunnerSimplifier(sub_inputs, do, save_path=tmp_path, n_process=8, verbose=verbose,
                                              return_results=True)
        outputs = mpr.run()
        os.remove(tmp_path)

        # Aggregate results
        tensor_outputs = []
        for output in outputs:
            ap, mrr, auc = json.loads(output)
            tensor_outputs.append([float(ap), float(mrr), float(auc)])

        tensor_outputs = torch.tensor(tensor_outputs, dtype=torch.float32, device=self.device)
        tensor_outputs = self.padded_gather(tensor_outputs)

        # Record results
        avg_results = {}
        for subsection in swissprot_subsections:
            avg_results[subsection] = {"map": [],
                                       "mrr": [],
                                       "auc": []}

        for input, output in zip(inputs, tensor_outputs):
            ap, mrr, auc = output
            subsection, _, _ = input

            avg_results[subsection]["map"].append(ap.cpu().item())
            avg_results[subsection]["mrr"].append(mrr.cpu().item())
            avg_results[subsection]["auc"].append(auc.cpu().item())

        results = {
            f"Text2{modality}_Total_mrr": np.mean(avg_results["Total"]["mrr"]),
            f"Text2{modality}_Total_map": np.mean(avg_results["Total"]["map"]),
            f"Text2{modality}_Total_auc": np.mean(avg_results["Total"]["auc"]),
        }

        # Average the precision and recall for each level
        for level, labels in [("residue-level", residue_level),
                              ("sequence-level", sequence_level),
                              ("all", residue_level | sequence_level)]:

            mrrs = []
            maps = []
            aucs = []
            for subsection in labels:
                if subsection in avg_results:
                    mrrs.append(np.mean(avg_results[subsection]["mrr"]))
                    maps.append(np.mean(avg_results[subsection]["map"]))
                    aucs.append(np.mean(avg_results[subsection]["auc"]))

            results[f"Text2{modality}_{level}_mrr"] = np.mean(mrrs)
            results[f"Text2{modality}_{level}_map"] = np.mean(maps)
            results[f"Text2{modality}_{level}_auc"] = np.mean(aucs)

        return results

    def retrieval_eval(self) -> dict:
        # Get protein representations
        protein_indices = self._get_protein_indices()
        
        # Get structure representations
        # if self.structure_config is not None:
        #     structure_embeddings = self._get_structure_embeddings()

        # Get text representations
        text_indices = self._get_text_indices()

        # Retrieve texts for each protein
        results = {}
        results.update(self._protein2text("Sequence", protein_indices, text_indices))
        # if self.structure_config is not None:
        #     results.update(self._protein2text("Structure", structure_embeddings, text_embeddings))
        #     results.update(self._text2protein("Structure", structure_embeddings, text_embeddings))
        
        # Retrieve proteins for each text
        results.update(self._text2protein("Sequence", protein_indices, text_indices))

        return results
    
    def _apply_bert_mask(self, tokens, tokenizer, mask_ratio):
        while True:
            masked_tokens = copy.copy(tokens)
            labels = torch.full((len(tokens) + 2,), -1, dtype=torch.long)
            vocab = [k for k in tokenizer.get_vocab().keys()]

            for i in range(len(tokens)):
                token = tokens[i]

                prob = random.random()
                if prob < mask_ratio:
                    prob /= mask_ratio
                    labels[i + 1] = tokenizer.convert_tokens_to_ids(token)

                    if prob < 0.8:
                        # 80% random change to mask token
                        if self.use_saprot:
                            token = "#" + token[-1]
                        else:
                            token = tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token
                        token = random.choice(vocab)
                    else:
                        # 10% chance to keep current token
                        pass

                    masked_tokens[i] = token

            # Check if there is at least one masked token
            if (labels != -1).any():
                return masked_tokens, labels
    
    def mlm_eval(self) -> float:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if self.use_saprot:
            proteins = []
            for sub_dict in self.uniprot2label.values():
                aa_seq = sub_dict["seq"]
                foldseek_seq = sub_dict["foldseek"]
                assert len(aa_seq) == len(foldseek_seq)
                seq = "".join([a + b for a, b in zip(aa_seq, foldseek_seq)])
                proteins.append(seq)

        else:
            proteins = [sub_dict["seq"] for sub_dict in self.uniprot2label.values()]

        span = math.ceil(len(proteins) / world_size)
        sub_proteins = proteins[rank * span: (rank + 1) * span]
        
        # Display the progress bar on the rank 0 process
        if self.trainer.local_rank == 0:
            iterator = tqdm(sub_proteins, desc="Computing mlm...")
        else:
            iterator = sub_proteins
        
        total = torch.tensor([0], dtype=torch.long, device=self.device)
        correct = torch.tensor([0], dtype=torch.long, device=self.device)
        for seq in iterator:
            tokens = self.protein_encoder.tokenizer.tokenize(seq)
            masked_tokens, labels = self._apply_bert_mask(tokens, self.protein_encoder.tokenizer, 0.15)
            seq = " ".join(masked_tokens)
            
            inputs = self.protein_encoder.tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            _, logits = self.protein_encoder(inputs, get_mask_logits=True)
            
            logits = logits.squeeze(0)
            labels = labels.to(self.device)
            
            selecor = labels != -1
            preds = logits.argmax(dim=-1)[selecor]
            labels = labels[selecor]
            
            total += len(preds)
            correct += (preds == labels).sum()
        
        # Gather all results
        total = self.padded_gather(total).sum()
        correct = self.padded_gather(correct).sum()
        
        acc = correct / total
        return acc.cpu().item()
    
    def _load_eval_data(self, stage):
        # Load the data
        lmdb_dir = eval(f"self.trainer.datamodule.{stage}_lmdb")
        uniprot2label_path = os.path.join(lmdb_dir, "uniprot2label.json")
        label2text_path = os.path.join(lmdb_dir, "label2text.json")
        swissprot_id_path = os.path.join(lmdb_dir, "swissprot_ids.tsv")
        
        self.uniprot2label = json.load(open(uniprot2label_path, "r"))
        self.label2text = json.load(open(label2text_path, "r"))
        self.swissprot_ids = set(pd.read_csv(swissprot_id_path, sep="\t", header=None).values.flatten().tolist())
        self.k = 3
    
    def on_test_start(self):
        self._load_eval_data("test")
        
        log_dict = self.retrieval_eval()
        log_dict = {"test_" + k: v for k, v in log_dict.items()}
        if self.use_mlm_loss:
            log_dict["test_mask_acc"] = self.mlm_eval()
        self.log_info(log_dict)
        print(log_dict)
    
    def on_validation_start(self):
        # Clear the cache
        torch.cuda.empty_cache()

        self._load_eval_data("valid")

        log_dict = self.retrieval_eval()
        log_dict = {"valid_" + k: v for k, v in log_dict.items()}
        if self.use_mlm_loss:
            log_dict["valid_mask_acc"] = self.mlm_eval()
        self.log_info(log_dict)

        self.check_save_condition(self.step, mode="max")
    
    def test_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        return
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # Re-sample the subset of the training data
        if self.trainer.datamodule.train_dataset.fixed_dataset_num is not None:
            self.trainer.datamodule.train_dataset.sample_subset()
    
    # def test_epoch_end(self, outputs):
    #     log_dict = self.get_log_dict("test")
    #     log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
    #
    #     print(log_dict)
    #     self.log_info(log_dict)
    #
    #     self.reset_metrics("test")
    #
    # def validation_epoch_end(self, outputs):
    #     log_dict = self.get_log_dict("valid")
    #     log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
    #
    #     self.log_info(log_dict)
    #     self.reset_metrics("valid")
    #     self.check_save_condition(log_dict["valid_loss"], mode="min")

