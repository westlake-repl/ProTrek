import gradio as gr
import torch
import pandas as pd

from utils.foldseek_util import get_struc_seq
from .init_model import model, all_index


# Samples for input
samples = [
    ["Proteins with zinc bindings."],
    ["Proteins locating at cell membrane."],
    ["Protein that serves as an enzyme."]
]

# Choices for subsection type
# valid_subsections = {"Function", "Subcellular location", "Protein names", "Sequence similarities", "GO annotation", "Global"}
valid_subsections = all_index["text"].keys()
# Sort the subsections
valid_subsections = sorted(valid_subsections)


def clear_results():
    return ""


# Search from database
def search(input: str, topk: int, input_type: str, query_type: str, subsection_type: str):
    input_modality = input_type.split(" ")[-1].replace("sequence", "protein")
    with torch.no_grad():
        input_embedding = getattr(model, f"get_{input_modality}_repr")([input]).cpu().numpy()
    
    output_modality = query_type.split(" ")[-1]
    if output_modality == "text":
        index = all_index["text"][subsection_type]["index"]
        ids = all_index["text"][subsection_type]["ids"]
        
    else:
        index = all_index[output_modality]["index"]
        ids = all_index[output_modality]["ids"]
        
    scores, ranks = index.search(input_embedding, topk)
    scores = scores / model.temperature.item()
    
    # Get topk ids
    topk_ids = []
    for rank in ranks[0]:
        now_id = ids[rank]
        if query_type == "text":
            topk_ids.append(now_id)
        else:
            # Provide link to uniprot website
            topk_ids.append(f"[{now_id}](https://www.uniprot.org/uniprotkb/{now_id})")
    
    df = pd.DataFrame({"Id": topk_ids, "Matching score": scores[0]})
    output = df.to_markdown()
    
    return output


def change_input_type(choice: str):
    # Change examples if input type is changed
    global samples
    if choice == "text":
        samples = [
            ["Proteins with zinc bindings."],
            ["Proteins locating at cell membrane."],
            ["Protein that serves as an enzyme."]
        ]

    elif choice == "protein sequence":
        samples = [
            ["MSATAEQNARNPKGKGGFARTVSQRKRKRLFLIGGALAVLAVAVGLMLTAFNQDIRFFRTPADLTEQDMTSGARFRLGGLVEEGSVSRTGSELRFTVTDTIKTVKVVFEGIPPDLFREGQGVVAEGRFGSDGLFRADNVLAKHDENYVPKDLADSLKKKGVWEGK"],
            ["MITLDWEKANGLITTVVQDATTKQVLMVAYMNQESLAKTMATGETWFWSRSRKTLWHKGATSGNIQTVKTIAVDCDADTLLVTVDPAGPACHTGHISCFYRHYPEGKDLT"],
            ["MDLKQYVSEVQDWPKPGVSFKDITTIMDNGEAYGYATDKIVEYAKDRDVDIVVGPEARGFIIGCPVAYSMGIGFAPVRKEGKLPREVIRYEYDLEYGTNVLTMHKDAIKPGQRVLITDDLLATGGTIEAAIKLVEKLGGIVVGIAFIIELKYLNGIEKIKDYDVMSLISYDE"]
        ]

    elif choice == "protein structure":
        samples = [
            ["dddddddddddddddpdpppvcppvnvvvvvvvvvvvvvvvvvvvvvvvvvvqdpqdedeqvrddpcqqpvqhkhkykafwappqwdddpqkiwtwghnppgiaieieghdappqddhrfikifiaghdpvrhtygdhidtdddpddddvvnvvvcvvvvndpdd"],
            ["dddadcpvpvqkakefeaeppprdtadiaiagpvqvvvcvvpqwhwgqdpvvrdidgqcpvpvqiwrwddwdaddnrryiytythtpahsdpvrhvhpppadvvgpddpd"],
            ["dplvvqwdwdaqpphhpdtdthcvscvvppvslvvqlvvvlvvcvvqvaqeeeeepdqrcsnrvsscvvvvhyywykyfpppddaawdwdwdddppgitiiithlpseaaageyeyegaeqalqprvlrvvvrcvvnnyddaeyeyqeyevcrvncvsvvvhhydyvyydpd"]
        ]
    
    # Set visibility of upload button
    if choice == "text":
        visible = False
    else:
        visible = True
        
    return samples, "", gr.update(visible=visible)


# Load example from dataset
def load_example(example_id):
    return samples[example_id][0]
 
 
# Change the visibility of subsection type
def subsection_visibility(query_type: str):
    if query_type == "text":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
 
 
# Parse the uploaded structure file
def parse_pdb_file(input_type, file):
    parsed_seqs = get_struc_seq("bin/foldseek", file)

    for seqs in parsed_seqs.values():
        if input_type == "protein sequence":
            return seqs[0]
        else:
            return seqs[1].lower()


# Build the block for text to protein
def build_search_module():
    gr.Markdown(f"# Search from Swiss-Prot database (the whole UniProt database will be supported soon)")
    with gr.Row(equal_height=True):
        with gr.Column():
            # Set input type
            input_type = gr.Radio(["protein sequence", "protein structure", "text"], label="Input type (e.g. 'text' means searching based on text descriptions)", value="text")
            
            with gr.Row():
                # Set query type
                query_type = gr.Radio(["protein sequence", "protein structure", "text"], label="Query type (e.g. 'protein sequence' means returning qualified protein sequences)", value="protein sequence")
            
                # If the query type is "text", provide an option to choose the subsection of text
                subsection_type = gr.Dropdown(list(valid_subsections), label="Subsection of text", value="Function",
                                              scale=0, interactive=True, visible=False)
                
                # Add event listener to query type
                query_type.change(fn=subsection_visibility, inputs=[query_type], outputs=[subsection_type])
            
            with gr.Row():
                # Input box
                input = gr.Text(label="Input")
                
                # Provide an upload button to upload a pdb file
                upload_btn = gr.UploadButton(label="Upload .pdb/.cif file", scale=0, visible=False)
                upload_btn.upload(parse_pdb_file, inputs=[input_type, upload_btn], outputs=[input])
            
            # Choose topk results
            topk = gr.Slider(1, 100, 5,  step=1, label="Retrieve top k results")

            # Provide examples
            examples = gr.Dataset(samples=samples, components=[input], type="index", label="Input examples")
            
            # Add click event to examples
            examples.click(fn=load_example, inputs=[examples], outputs=input)
            
            # Change examples based on input type
            input_type.change(fn=change_input_type, inputs=[input_type], outputs=[examples, input, upload_btn])
            
            with gr.Row():
                t2p_btn = gr.Button(value="Search")
                clear_btn = gr.Button(value="Clear")
        
        results = gr.Markdown(label="results")
        t2p_btn.click(fn=search, inputs=[input, topk, input_type, query_type, subsection_type], outputs=results)
        clear_btn.click(fn=clear_results, outputs=results)