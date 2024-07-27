import gradio as gr
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm
from .init_model import model, all_index, valid_subsections
from .blocks import upload_pdb_button, parse_pdb_file


tmp_file_path = "/tmp/results.tsv"
tmp_plot_path = "/tmp/histogram.svg"

# Samples for input
samples = [
    ["Proteins with zinc bindings."],
    ["Proteins locating at cell membrane."],
    ["Protein that serves as an enzyme."]
]

# Databases for different modalities
now_db = {
    "sequence": list(all_index["sequence"].keys())[0],
    "structure": list(all_index["structure"].keys())[0],
    "text": list(all_index["text"].keys())[0]
}


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
    _, ymax = plt.ylim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p)
    
    # Plot total number of scores
    plt.text(xmax, 0.9*ymax, f"Total number: {len(scores)}", ha='right', fontsize=12)

    # Convert the plot to svg format
    plt.savefig(tmp_plot_path)
    plt.cla()


# Search from database
def search(input: str, nprobe: int, topk: int, input_type: str, query_type: str, subsection_type: str):
    input_modality = input_type.replace("sequence", "protein")
    with torch.no_grad():
        input_embedding = getattr(model, f"get_{input_modality}_repr")([input]).cpu().numpy()

    db = now_db[query_type]
    if query_type == "text":
        index = all_index["text"][db][subsection_type]["index"]
        ids = all_index["text"][db][subsection_type]["ids"]

    else:
        index = all_index[query_type][db]["index"]
        ids = all_index[query_type][db]["ids"]
        
    if check_index_ivf(query_type, subsection_type):
        if index.nlist < nprobe:
            raise gr.Error(f"The number of clusters to search must be less than or equal to the number of clusters in the index ({index.nlist}).")
        else:
            index.nprobe = nprobe
    
    if topk > index.ntotal:
        raise gr.Error(f"You cannot retrieve more than the database size ({index.ntotal}).")
    
    # Retrieve all scores to plot the distribution
    scores, ranks = index.search(input_embedding, index.ntotal)
    scores, ranks = scores[0], ranks[0]
    
    # Remove inf values
    selector = scores > -1
    scores = scores[selector]
    ranks = ranks[selector]
    scores = scores / model.temperature.item()
    plot(scores)
    
    top_scores = scores[:topk]
    top_ranks = ranks[:topk]
    
    # ranks = [list(range(topk))]
    # ids = ["P12345"] * topk
    # scores = torch.randn(topk).tolist()
    
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
            if db != "PDB":
                # Provide link to uniprot website
                topk_ids.append(f"[{now_id}](https://www.uniprot.org/uniprotkb/{now_id})")
            else:
                # Provide link to pdb website
                pdb_id = now_id.split("-")[0]
                topk_ids.append(f"[{now_id}](https://www.rcsb.org/structure/{pdb_id})")
    
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

    elif choice == "sequence":
        samples = [
            ["MSATAEQNARNPKGKGGFARTVSQRKRKRLFLIGGALAVLAVAVGLMLTAFNQDIRFFRTPADLTEQDMTSGARFRLGGLVEEGSVSRTGSELRFTVTDTIKTVKVVFEGIPPDLFREGQGVVAEGRFGSDGLFRADNVLAKHDENYVPKDLADSLKKKGVWEGK"],
            ["MITLDWEKANGLITTVVQDATTKQVLMVAYMNQESLAKTMATGETWFWSRSRKTLWHKGATSGNIQTVKTIAVDCDADTLLVTVDPAGPACHTGHISCFYRHYPEGKDLT"],
            ["MDLKQYVSEVQDWPKPGVSFKDITTIMDNGEAYGYATDKIVEYAKDRDVDIVVGPEARGFIIGCPVAYSMGIGFAPVRKEGKLPREVIRYEYDLEYGTNVLTMHKDAIKPGQRVLITDDLLATGGTIEAAIKLVEKLGGIVVGIAFIIELKYLNGIEKIKDYDVMSLISYDE"]
        ]

    elif choice == "structure":
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
def change_output_type(query_type: str, subsection_type: str):
    nprobe_visible = check_index_ivf(query_type, subsection_type)
    subsection_visible = True if query_type == "text" else False
    
    return (
        gr.update(visible=subsection_visible), 
        gr.update(visible=nprobe_visible),
        gr.update(choices=list(all_index[query_type].keys()), value=now_db[query_type])
    )


def check_index_ivf(index_type: str, subsection_type: str = None) -> bool:
    """
    Check if the index is of IVF type.
    Args:
        index_type: Type of index.
        subsection_type: If the "index_type" is "text", get the index based on the subsection type.

    Returns:
        Whether the index is of IVF type or not.
    """
    db = now_db[index_type]
    if index_type == "sequence":
        index = all_index["sequence"][db]["index"]
    
    elif index_type == "structure":
        index = all_index["structure"][db]["index"]
    
    elif index_type == "text":
        index = all_index["text"][db][subsection_type]["index"]
    
    nprobe_visible = True if hasattr(index, "nprobe") else False
    return nprobe_visible


def change_db_type(query_type: str, subsection_type: str, db_type: str):
    """
    Change the database to search.
    Args:
        query_type: The output type.
        db_type: The database to search.
    """
    now_db[query_type] = db_type
    
    if query_type == "text":
        subsection_update = gr.update(choices=list(valid_subsections[now_db["text"]]), value="Function")
    else:
        subsection_update = gr.update(visible=False)
    
    nprobe_visible = check_index_ivf(query_type, subsection_type)
    return subsection_update, gr.update(visible=nprobe_visible)


# Build the searching block
def build_search_module():
    gr.Markdown(f"# Search from Swiss-Prot database (the whole UniProt database will be supported soon)")
    with gr.Row(equal_height=True):
        with gr.Column():
            # Set input type
            input_type = gr.Radio(["sequence", "structure", "text"], label="Input type (e.g. 'text' means searching based on text descriptions)", value="text")

            with gr.Row():
                # Set output type
                query_type = gr.Radio(
                    ["sequence", "structure", "text"],
                    label="Output type (e.g. 'sequence' means returning qualified sequences)",
                    value="sequence",
                    scale=2,
                )
            
                # If the output type is "text", provide an option to choose the subsection of text
                subsection_type = gr.Dropdown(valid_subsections[now_db["text"]], label="Subsection of text", value="Function",
                                              interactive=True, visible=False, scale=0)
                
                db_type = gr.Dropdown(all_index["sequence"].keys(), label="Database", value=now_db["sequence"],
                                              interactive=True, visible=True, scale=0)

            with gr.Row():
                # Input box
                input = gr.Text(label="Input")
                
                # Provide an upload button to upload a pdb file
                upload_btn, chain_box = upload_pdb_button(visible=False)
                upload_btn.upload(parse_pdb_file, inputs=[input_type, upload_btn, chain_box], outputs=[input])
            
            
            # If the index is of IVF type, provide an option to choose the number of clusters.
            nprobe_visible = check_index_ivf(query_type.value)
            nprobe = gr.Slider(1, 1000000, 1000,  step=1, visible=nprobe_visible,
                               label="Number of clusters to search (lower value for faster search and higher value for more accurate search)")
            
            # Add event listener to output type
            query_type.change(fn=change_output_type, inputs=[query_type, subsection_type],
                              outputs=[subsection_type, nprobe, db_type])
            
            # Add event listener to db type
            db_type.change(fn=change_db_type, inputs=[query_type, subsection_type, db_type],
                           outputs=[subsection_type, nprobe])
            
            # Choose topk results
            topk = gr.Slider(1, 1000000, 5,  step=1, label="Retrieve top k results")

            # Provide examples
            examples = gr.Dataset(samples=samples, components=[input], type="index", label="Input examples")
            
            # Add click event to examples
            examples.click(fn=load_example, inputs=[examples], outputs=input)
            
            # Change examples based on input type
            input_type.change(fn=change_input_type, inputs=[input_type], outputs=[examples, input, upload_btn, chain_box])
            
            with gr.Row():
                search_btn = gr.Button(value="Search")
                clear_btn = gr.Button(value="Clear")
        
        with gr.Row():
            with gr.Column():
                results = gr.Markdown(label="results", height=450)
                download_btn = gr.DownloadButton(label="Download results", visible=False)
            
                # Plot the distribution of scores
                histogram = gr.Image(label="Histogram of matching scores", type="filepath", scale=1, visible=False)
            
        search_btn.click(fn=search, inputs=[input, nprobe, topk, input_type, query_type, subsection_type],
                      outputs=[results, download_btn, histogram])
        
        clear_btn.click(fn=clear_results, outputs=[results, download_btn, histogram])