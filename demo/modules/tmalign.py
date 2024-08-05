import gradio as gr
import os

from .blocks import upload_pdb_button
from utils.downloader import download_pdb, download_af2


root_dir = __file__.rsplit("/", 3)[0]
structure_types = ["AlphaFoldDB", "PDB"]


def upload_structure(file: str):
    return file


def get_structure_path(structure: str, structure_type: str) -> str:
    # If the structure is manually uploaded
    if structure[0] == "/":
        return structure
    
    # If the structure is a Uniprot ID, download the structure from AlphaFoldDB
    elif structure_type == "AlphaFoldDB":
        save_path = f"{root_dir}/demo/cache/{structure}.pdb"
        if not os.path.exists(save_path):
            download_af2(structure, "pdb", save_path)
        return save_path
    
    # If the structure is a PDB ID, download the structure from PDB
    elif structure_type == "PDB":
        save_path = f"{root_dir}/demo/cache/{structure}.cif"
        if not os.path.exists(save_path):
            download_pdb(structure, "cif", save_path)
        return save_path
    
    
def tmalign(structure_1: str, structure_type_1: str, structure_2: str, structure_type_2: str):
    structure_path_1 = get_structure_path(structure_1, structure_type_1)
    structure_path_2 = get_structure_path(structure_2, structure_type_2)
    
    cmd = f"bin/TMalign {structure_path_1} {structure_path_2}"
    
    r = os.popen(cmd)
    text = r.read()
    return text


# Build the block for computing protein-text similarity
def build_TMalign():
    gr.Markdown(f"# Calculate TM-score between two protein structures")
    with gr.Row(equal_height=True):
        with gr.Column():
            # Compute similarity score between sequence and text
            with gr.Row():
                structure_1 = gr.Textbox(label="Protein structure 1 (input Uniprot ID or PDB ID or upload a pdb file)")
                
                structure_type_1 = gr.Dropdown(structure_types, label="Structure type (if the structure is manually uploaded, ignore this field)",
                                               value="AlphaFoldDB", interactive=True, visible=True)
    
                # Provide an upload button to upload a pdb file
                upload_btn_1, _ = upload_pdb_button(visible=True, chain_visible=False)
                upload_btn_1.upload(upload_structure, inputs=[upload_btn_1], outputs=[structure_1])
            
            with gr.Row():
                structure_2 = gr.Textbox(label="Protein structure 2 (input Uniprot ID or PDB ID or upload a pdb file)")
                
                structure_type_2 = gr.Dropdown(structure_types, label="Structure type (if the structure is manually uploaded, ignore this field)",
                                               value="AlphaFoldDB", interactive=True, visible=True)
                
                # Provide an upload button to upload a pdb file
                upload_btn_2, _ = upload_pdb_button(visible=True, chain_visible=False)
                upload_btn_2.upload(upload_structure, inputs=[upload_btn_2], outputs=[structure_2])
            
            compute_btn = gr.Button(value="Compute TM-score")
            tmscore = gr.TextArea(label="TM-score", interactive=False)
            
            compute_btn.click(tmalign, inputs=[structure_1, structure_type_1, structure_2, structure_type_2],
                              outputs=[tmscore])
            