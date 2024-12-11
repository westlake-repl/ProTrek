import gradio as gr

from utils.foldseek_util import get_struc_seq


ROOT_DIR = __file__.rsplit("/", 3)[0]


####################################################
#                  gradio blocks                   #
####################################################
def upload_pdb_button(visible: bool = True, chain_visible: bool = True):
    """
    Provide an upload button to upload a pdb file
    Args:
        visible: Whether the block is visible or not
    """
    
    with gr.Column(scale=0):
        
        # Which chain to be extracted
        chain_box = gr.Textbox(label="Chain (to be extracted from the pdb file)", value="A",
                               visible=chain_visible, interactive=True)
        
        upload_btn = gr.UploadButton(label="Upload .pdb/.cif file", visible=visible)
        
    return upload_btn, chain_box


####################################################
#                 Trigger functions                #
####################################################
def parse_pdb_file(input_type: str, file: str, chain: str) -> str:
    """
    Parse the uploaded structure file
    
    Args:
        input_type: Type of input. Must be one of ["protein sequence", "protein structure"]
        
        file: Path to the uploaded file
        
        chain: Chain to be extracted from the pdb file

    Returns:
        Protein sequence or Foldseek sequence
    """
    try:
        parsed_seqs = get_struc_seq(f"{ROOT_DIR}/bin/foldseek", file, [chain])[chain]
        if input_type == "sequence":
            return parsed_seqs[0]
        else:
            return parsed_seqs[1].lower()
    
    except KeyError:
        raise gr.Error(f"Chain '{chain}' not found in the pdb file. Please check the chain id and try again.")
    
    except Exception as e:
        raise gr.Error(str(e))


def set_upload_visible(visible: bool) -> gr.Interface:
    """
    Set the visibility of the upload button
    
    Args:
        visible: Whether the block is visible or not
    
    Returns:
        gr.Interface: Updated interface
    """
    
    return gr.update(visible=visible)
