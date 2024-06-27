import gradio as gr
import torch
import pandas as pd

from .init_model import model, all_index
from .blocks import upload_pdb_button, parse_pdb_file


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
    return "", gr.update(visible=False)


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

    # Write the results to a temporary file for downloading
    path = f"/tmp/results.tsv"
    with open(path, "w") as w:
        w.write("Id\tMatching score\n")
        for i in range(topk):
            w.write(f"{ids[i]}\t{scores[0][i]}\n")
    
    assert topk < index.ntotal, "You cannot retrieve more than the database size."
    
    # Get topk ids
    topk_ids = []
    for rank in ranks[0]:
        now_id = ids[rank]
        if query_type == "text":
            topk_ids.append(now_id)
        else:
            # Provide link to uniprot website
            topk_ids.append(f"[{now_id}](https://www.uniprot.org/uniprotkb/{now_id})")
    
    limit = 1000
    df = pd.DataFrame({"Id": topk_ids[:limit], "Matching score": scores[0][:limit]})
    if len(topk_ids) > limit:
        info_df = pd.DataFrame({"Id": ["Download the file to check all results"], "Matching score": ["..."]},
                               index=[1000])
        df = pd.concat([df, info_df], axis=0)
    
    output = df.to_markdown()
    return output, gr.DownloadButton(label="Download results", value=path, visible=True, scale=0)


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
    
    return samples, "", gr.update(visible=visible), gr.update(visible=visible)


# Load example from dataset
def load_example(example_id):
    return samples[example_id][0]
 
 
# Change the visibility of subsection type
def subsection_visibility(query_type: str):
    if query_type == "text":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
 

# Build the block for text to protein
def build_search_module():
    gr.Markdown(f"# Search from Swiss-Prot database (the whole UniProt database will be supported soon)")
    with gr.Row(equal_height=True):
        with gr.Column():
            # Set input type
            input_type = gr.Radio(["protein sequence", "protein structure", "text"], label="Input type (e.g. 'text' means searching based on text descriptions)", value="text")
            
            with gr.Row():
                # Set output type
                query_type = gr.Radio(["protein sequence", "protein structure", "text"], label="Output type (e.g. 'protein sequence' means returning qualified protein sequences)", value="protein sequence")
            
                # If the output type is "text", provide an option to choose the subsection of text
                subsection_type = gr.Dropdown(list(valid_subsections), label="Subsection of text", value="Function",
                                              scale=0, interactive=True, visible=False)
                
                # Add event listener to output type
                query_type.change(fn=subsection_visibility, inputs=[query_type], outputs=[subsection_type])
            
            with gr.Row():
                # Input box
                input = gr.Text(label="Input")
                
                # Provide an upload button to upload a pdb file
                upload_btn, chain_box = upload_pdb_button(visible=False)
                upload_btn.upload(parse_pdb_file, inputs=[input_type, upload_btn, chain_box], outputs=[input])
            
            # Choose topk results
            topk = gr.Slider(1, 1000000, 5,  step=1, label="Retrieve top k results")

            # Provide examples
            examples = gr.Dataset(samples=samples, components=[input], type="index", label="Input examples")
            
            # Add click event to examples
            examples.click(fn=load_example, inputs=[examples], outputs=input)
            
            # Change examples based on input type
            input_type.change(fn=change_input_type, inputs=[input_type], outputs=[examples, input, upload_btn, chain_box])
            
            with gr.Row():
                t2p_btn = gr.Button(value="Search")
                clear_btn = gr.Button(value="Clear")
        
        with gr.Column():
            results = gr.Markdown(label="results", height=450)
            download_btn = gr.DownloadButton(label="Download results", visible=False)
            
        t2p_btn.click(fn=search, inputs=[input, topk, input_type, query_type, subsection_type],
                      outputs=[results, download_btn])
        
        clear_btn.click(fn=clear_results, outputs=[results, download_btn])