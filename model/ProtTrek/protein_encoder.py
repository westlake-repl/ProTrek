import torch

from tqdm import tqdm
from torch.nn.functional import normalize
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer


class ProteinEncoder(torch.nn.Module):
    def __init__(self,
                 config_path: str,
                 out_dim: int,
                 load_pretrained: bool = True,
                 gradient_checkpointing: bool = False):
        """
        Args:
            config_path: Path to the config file
            
            out_dim    : Output dimension of the protein representation
            
            load_pretrained: Whether to load pretrained weights
            
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        config = EsmConfig.from_pretrained(config_path)
        if load_pretrained:
            self.model = EsmForMaskedLM.from_pretrained(config_path)
        else:
            self.model = EsmForMaskedLM(config)
        self.out = torch.nn.Linear(config.hidden_size, out_dim)
        
        # Set gradient checkpointing
        self.model.esm.encoder.gradient_checkpointing = gradient_checkpointing
        
        # Remove contact head
        self.model.esm.contact_head = None
        
        # Remove position embedding if the embedding type is ``rotary``
        if config.position_embedding_type == "rotary":
            self.model.esm.embeddings.position_embeddings = None
        
        self.tokenizer = EsmTokenizer.from_pretrained(config_path)
    
    def get_repr(self, proteins: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        """
        Compute protein representation for the given proteins
        Args:
            protein: A list of protein sequences
            batch_size: Batch size for inference
            verbose: Whether to print progress
        """
        device = next(self.parameters()).device
        
        protein_repr = []
        if verbose:
            iterator = tqdm(range(0, len(proteins), batch_size), desc="Computing protein embeddings")
        else:
            iterator = range(0, len(proteins), batch_size)
            
        for i in iterator:
            protein_inputs = self.tokenizer.batch_encode_plus(proteins[i:i + batch_size],
                                                              return_tensors="pt",
                                                              padding=True)
            protein_inputs = {k: v.to(device) for k, v in protein_inputs.items()}
            output, _ = self.forward(protein_inputs)
            
            protein_repr.append(output)
        
        protein_repr = torch.cat(protein_repr, dim=0)
        return normalize(protein_repr, dim=-1)

    def forward(self, inputs: dict, get_mask_logits: bool = False):
        """
        Encode protein sequence into protein representation
        Args:
            inputs: A dictionary containing the following keys:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
            get_mask_logits: Whether to return the logits for masked tokens

        Returns:
            protein_repr: [batch, protein_repr_dim]
            mask_logits : [batch, seq_len, vocab_size]
        """
        last_hidden_state = self.model.esm(**inputs).last_hidden_state
        reprs = last_hidden_state[:, 0, :]
        reprs = self.out(reprs)

        # Get logits for masked tokens
        if get_mask_logits:
            mask_logits = self.model.lm_head(last_hidden_state)
        else:
            mask_logits = None

        return reprs, mask_logits