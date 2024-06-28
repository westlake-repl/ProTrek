import gradio as gr
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm
from .init_model import model, all_index
from .blocks import upload_pdb_button, parse_pdb_file


tmp_file_path = "/tmp/results.tsv"
tmp_plot_path = "/tmp/histogram.svg"

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
    return "", gr.update(visible=False), gr.update(visible=False)


def plot(scores) -> None:
    """
    Plot the distribution of scores and fit a normal distribution.
    Args:
        scores: List of scores
    """
    plt.hist(scores, bins=100, density=True, alpha=0.6)
    plt.title('Distribution of similarity scores in the database', fontsize=15)
    plt.xlabel('Similarity score', fontsize=15)
    plt.ylabel('Density', fontsize=15)

    mu, std = norm.fit(scores)

    # Plot the Gaussian
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p)

    # Convert the plot to svg format
    plt.savefig(tmp_plot_path)
    plt.cla()


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
        
    scores, ranks = index.search(input_embedding, index.ntotal)
    scores, ranks = scores[0], ranks[0]
    scores = scores / model.temperature.item()
    plot(scores)

    top_scores = scores[:topk]
    top_ranks = ranks[:topk]
    
    # ranks = [list(range(topk))]
    # ids = ["P12345"] * topk
    # scores = torch.randn(topk).tolist()
    
    if topk > index.ntotal:
        raise gr.Error(f"You cannot retrieve more than the database size ({index.ntotal}).")

    # Write the results to a temporary file for downloading
    with open(tmp_file_path, "w") as w:
        w.write("Id\tMatching score\n")
        for i in range(topk):
            rank = top_ranks[i]
            w.write(f"{ids[rank]}\t{top_scores[i]}\n")
    
    # Get topk ids
    topk_ids = []
    for rank in top_ranks:
        now_id = ids[rank]
        if query_type == "text":
            topk_ids.append(now_id)
        else:
            # Provide link to uniprot website
            topk_ids.append(f"[{now_id}](https://www.uniprot.org/uniprotkb/{now_id})")
    
    limit = 1000
    df = pd.DataFrame({"Id": topk_ids[:limit], "Matching score": top_scores[:limit]})
    if len(topk_ids) > limit:
        info_df = pd.DataFrame({"Id": ["Download the file to check all results"], "Matching score": ["..."]},
                               index=[1000])
        df = pd.concat([df, info_df], axis=0)
    
    output = df.to_markdown()
    return (output,
            gr.DownloadButton(label="Download results", value=tmp_file_path, visible=True, scale=0),
            gr.update(value=tmp_plot_path, visible=True))


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
    
    return gr.update(samples=samples), "", gr.update(visible=visible), gr.update(visible=visible)


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
        
        with gr.Row():
            with gr.Column():
                results = gr.Markdown(label="results", height=450)
                download_btn = gr.DownloadButton(label="Download results", visible=False)
            
                # Plot the distribution of scores
                histogram = gr.Image(label="Histogram of matching scores", type="filepath", scale=1, visible=False)
            
        t2p_btn.click(fn=search, inputs=[input, topk, input_type, query_type, subsection_type],
                      outputs=[results, download_btn, histogram])
        
        clear_btn.click(fn=clear_results, outputs=[results, download_btn, histogram])