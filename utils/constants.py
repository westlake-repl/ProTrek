import itertools


aa_set = {"A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"}
aa_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

struc_unit = "abcdefghijklmnopqrstuvwxyz"


def create_vocab(size: int) -> dict:
    """

    Args:
        size:   Size of the vocabulary

    Returns:
        vocab:  Vocabulary
    """

    token_len = 1
    while size > len(struc_unit) ** token_len:
        token_len += 1

    vocab = {}
    for i, token in enumerate(itertools.product(struc_unit, repeat=token_len)):
        vocab[i] = "".join(token)
        if len(vocab) == size:
            vocab[i+1] = "#"
            return vocab

# ProTrek
residue_level = {"Active site", "Binding site", "Site", "DNA binding", "Natural variant", "Mutagenesis",
                 "Transmembrane", "Topological domain", "Intramembrane", "Signal peptide", "Propeptide",
                 "Transit peptide",
                 "Chain", "Peptide", "Modified residue", "Lipidation", "Glycosylation", "Disulfide bond",
                 "Cross-link",
                 "Domain", "Repeat", "Compositional bias", "Region", "Coiled coil", "Motif"}

sequence_level = {"Function", "Miscellaneous", "Caution", "Catalytic activity", "Cofactor", "Activity regulation",
                  "Biophysicochemical properties", "Pathway", "Involvement in disease", "Allergenic properties",
                  "Toxic dose", "Pharmaceutical use", "Disruption phenotype", "Subcellular location",
                  "Post-translational modification", "Subunit", "Domain (non-positional annotation)",
                  "Sequence similarities", "RNA Editing", "Tissue specificity", "Developmental stage", "Induction",
                  "Biotechnology", "Polymorphism", "GO annotation", "Proteomes", "Protein names", "Gene names",
                  "Organism", "Taxonomic lineage", "Virus host"}

raw_text_level = {"Function", "Subunit", "Tissue specificity", "Disruption phenotype", "Post-translational modification",
                  "Induction", "Miscellaneous", "Sequence similarities", "Developmental stage",
                  "Domain (non-positional annotation)", "Activity regulation", "Caution", "Polymorphism", "Toxic dose",
                  "Allergenic properties", "Pharmaceutical use", "Cofactor", "Biophysicochemical properties",
                  "Subcellular location", "RNA Editing"}