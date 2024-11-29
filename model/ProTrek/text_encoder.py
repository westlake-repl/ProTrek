import torch

from tqdm import tqdm
from torch.nn.functional import normalize
from transformers import BertConfig, BertModel, BertTokenizer


class TextEncoder(torch.nn.Module):
    def __init__(self,
                 config_path: str,
                 out_dim: int,
                 load_pretrained: bool = True,
                 gradient_checkpointing: bool = False):
        """
        Args:
            config_path: Path to the config file
            
            out_dim: Output dimension of the text representation
            
            load_pretrained: Whether to load pretrained weights
            
            gradient_checkpointing: Whether to enable gradient checkpointing
        """
        super().__init__()
        config = BertConfig.from_pretrained(config_path)
        if load_pretrained:
            self.model = BertModel.from_pretrained(config_path, add_pooling_layer=False)
        else:
            self.model = BertModel(config, add_pooling_layer=False)
        self.out = torch.nn.Linear(config.hidden_size, out_dim)
        
        # Set gradient checkpointing
        self.model.encoder.gradient_checkpointing = gradient_checkpointing
        
        self.tokenizer = BertTokenizer.from_pretrained(config_path)
    
    def get_repr(self, texts: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        """
        Compute text representation for the given texts
        Args:
            texts: A list of strings
            batch_size: Batch size for inference
            verbose: Whether to print progress
        """
        device = next(self.parameters()).device
        
        if isinstance(texts, str):
            texts = [texts]
        
        text_repr = []
        if verbose:
            iterator = tqdm(range(0, len(texts), batch_size), desc="Computing text embeddings")
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            text_inputs = self.tokenizer.batch_encode_plus(texts[i: i+batch_size],
                                                           return_tensors="pt",
                                                           truncation=True,
                                                           max_length=512,
                                                           padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            output = self(text_inputs)
            
            text_repr.append(output)
        
        text_repr = torch.cat(text_repr, dim=0)
        return normalize(text_repr, dim=-1)
    
    def forward(self, inputs: dict):
        """
        Encode text into text representation
        Args:
            inputs: A dictionary containing the following keys:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
                - token_type_ids: [batch, seq_len]

        Returns:
            text_repr: [batch, text_repr_dim]
        """
        reprs = self.model(**inputs).last_hidden_state[:, 0, :]
        reprs = self.out(reprs)
        return reprs