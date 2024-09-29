import gradio as gr
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm
from .init_model import model, all_index, valid_subsections
from .blocks import upload_pdb_button, parse_pdb_file
from Bio.Align import PairwiseAligner


tmp_file_path = "/tmp/results.tsv"
tmp_plot_path = "/tmp/histogram.svg"

# Samples for input
samples = {
    "sequence": [
            ["MSATAEQNARNPKGKGGFARTVSQRKRKRLFLIGGALAVLAVAVGLMLTAFNQDIRFFRTPADLTEQDMTSGARFRLGGLVEEGSVSRTGSELRFTVTDTIKTVKVVFEGIPPDLFREGQGVVAEGRFGSDGLFRADNVLAKHDENYVPKDLADSLKKKGVWEGK"],
            ["MITLDWEKANGLITTVVQDATTKQVLMVAYMNQESLAKTMATGETWFWSRSRKTLWHKGATSGNIQTVKTIAVDCDADTLLVTVDPAGPACHTGHISCFYRHYPEGKDLT"],
            ["MDLKQYVSEVQDWPKPGVSFKDITTIMDNGEAYGYATDKIVEYAKDRDVDIVVGPEARGFIIGCPVAYSMGIGFAPVRKEGKLPREVIRYEYDLEYGTNVLTMHKDAIKPGQRVLITDDLLATGGTIEAAIKLVEKLGGIVVGIAFIIELKYLNGIEKIKDYDVMSLISYDE"]
        ],

    "structure": [
            ["dddddddddddddddpdpppvcppvnvvvvvvvvvvvvvvvvvvvvvvvvvvqdpqdedeqvrddpcqqpvqhkhkykafwappqwdddpqkiwtwghnppgiaieieghdappqddhrfikifiaghdpvrhtygdhidtdddpddddvvnvvvcvvvvndpdd"],
            ["dddadcpvpvqkakefeaeppprdtadiaiagpvqvvvcvvpqwhwgqdpvvrdidgqcpvpvqiwrwddwdaddnrryiytythtpahsdpvrhvhpppadvvgpddpd"],
            ["dplvvqwdwdaqpphhpdtdthcvscvvppvslvvqlvvvlvvcvvqvaqeeeeepdqrcsnrvsscvvvvhyywykyfpppddaawdwdwdddppgitiiithlpseaaageyeyegaeqalqprvlrvvvrcvvnnyddaeyeyqeyevcrvncvsvvvhhydyvyydpd"]
        ],

    "text": [
        ["Proteins with zinc bindings."],
        ["Proteins locating at cell membrane."],
        ["Protein that serves as an enzyme."]
    ],
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
    y_ed = plt.gca().get_ylim()[-1]
    plt.ylim(-0.05, y_ed)

    # Add note
    x_st = plt.gca().get_xlim()[0]
    text = ("Note: For the \"UniRef50\" and \"Uncharacterized\" databases, the figure illustrates\n "
            "only top-ranked clusters (identified using Faiss), whereas for other databases, it\n "
            "displays the distribution across all samples.")
    plt.text(x_st, -0.04, text, fontsize=8)
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


# Calculate protein sequence identity
def calc_seq_identity(seq1: str, seq2: str) -> float:
    aligner = PairwiseAligner()
    aligner.mode = "local"

    alignment = next(aligner.align(seq1, seq2))
    a1, a2 = alignment
    identity = sum(1 for a, b in zip(a1, a2) if a == b) / len(a1)
    return identity


# Search from database
def search(input: str, nprobe: int, topk: int, input_type: str, query_type: str, subsection_type: str, db: str):
    print(f"Input type: {input_type}\n Output type: {query_type}\nDatabase: {db}\nSubsection: {subsection_type}")

    input_modality = input_type.replace("sequence", "protein")
    with torch.no_grad():
        input_embedding = getattr(model, f"get_{input_modality}_repr")([input]).cpu().numpy()

    if query_type == "text":
        index = all_index["text"][db][subsection_type]["index"]
        ids = all_index["text"][db][subsection_type]["ids"]

    else:
        index = all_index[query_type][db]["index"]
        ids = all_index[query_type][db]["ids"]

    if hasattr(index, "nprobe"):
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
        if query_type == "text":
            w.write("Id\tMatching score\n")
        else:
            w.write("Id\tSequence\tLength\tMatching score\n")

        for i in range(topk):
            rank = top_ranks[i]
            if query_type == "text":
                w.write(f"{ids.get(rank)}\t{top_scores[i]}\n")
            else:
                id, seq, length = ids.get(rank).split("\t")
                w.write(f"{id}\t{seq}\t{length}\t{top_scores[i]}\n")

    # Get topk ids
    topk_ids = []
    topk_seqs = []
    topk_lengths = []
    for rank in top_ranks:
        now_id = ids.get(rank).split("\t")[0]
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

        if query_type != "text":
            _, ori_seq, ori_len = ids.get(rank).split("\t")
            seq = ori_seq[:20] + "..." if len(ori_seq) > 20 else ori_seq
            topk_seqs.append(seq)
            topk_lengths.append(ori_len)

    # If both the input and output are protein sequences, calculate the sequence identity
    if input_type == "sequence" and query_type == "sequence":
        seq_identities = [calc_seq_identity(input, ids.get(rank).split("\t")[1]) for rank in top_ranks]
        seq_identities = [f"{identity*100:.2f}%" for identity in seq_identities]

    limit = 1000
    if query_type == "text":
        df = pd.DataFrame({"Id": topk_ids[:limit], "Matching score": top_scores[:limit]})
        if len(topk_ids) > limit:
            info_df = pd.DataFrame({"Id": ["Download the file to check all results"], "Matching score": ["..."]},
                                   index=[1000])
            df = pd.concat([df, info_df], axis=0)

    elif input_type == "sequence" and query_type == "sequence":
        df = pd.DataFrame({"Id": topk_ids[:limit], "Sequence": topk_seqs[:limit],
                           "Length": topk_lengths[:limit], "Sequence identity": seq_identities[:limit],
                          "Matching score": top_scores[:limit]})
        if len(topk_ids) > limit:
            info_df = pd.DataFrame({"Id": ["Download the file to check all results"], "Sequence": ["..."], "Length": ["..."],
                                    "Sequence identity": ["..."], "Matching score": ["..."]}, index=[1000])
            df = pd.concat([df, info_df], axis=0)

    else:
        df = pd.DataFrame({"Id": topk_ids[:limit], "Sequence": topk_seqs[:limit], "Length": topk_lengths[:limit], "Matching score": top_scores[:limit]})
        if len(topk_ids) > limit:
            info_df = pd.DataFrame({"Id": ["Download the file to check all results"], "Sequence": ["..."], "Length": ["..."], "Matching score": ["..."]},
                                   index=[1000])
            df = pd.concat([df, info_df], axis=0)
    
    output = df.to_markdown()
    return (output,
            gr.DownloadButton(label="Download results", value=tmp_file_path, visible=True, scale=0),
            gr.update(value=tmp_plot_path, visible=True))


def change_input_type(choice: str):
    # Change examples if input type is changed
    global samples
    
    # Set visibility of upload button
    if choice == "text":
        visible = False
    else:
        visible = True
    
    return gr.update(samples=samples[choice]), "", gr.update(visible=visible), gr.update(visible=visible)


# Load example from dataset
def load_example(example_id):
    return example_id[0]
 
 
# Change the visibility of subsection type
def change_output_type(query_type: str, subsection_type: str):
    db_type = list(all_index[query_type].keys())[0]
    nprobe_visible = check_index_ivf(query_type, db_type, subsection_type)
    subsection_visible = True if query_type == "text" else False

    return (
        gr.update(visible=subsection_visible),
        gr.update(visible=nprobe_visible),
        gr.update(choices=list(all_index[query_type].keys()), value=db_type)
    )


def check_index_ivf(index_type: str, db: str, subsection_type: str = None) -> bool:
    """
    Check if the index is of IVF type.
    Args:
        index_type: Type of index.
        subsection_type: If the "index_type" is "text", get the index based on the subsection type.

    Returns:
        Whether the index is of IVF type or not.
    """
    if index_type == "sequence":
        index = all_index["sequence"][db]["index"]
    
    elif index_type == "structure":
        index = all_index["structure"][db]["index"]
    
    elif index_type == "text":
        index = all_index["text"][db][subsection_type]["index"]
    
    # nprobe_visible = True if hasattr(index, "nprobe") else False
    # return nprobe_visible
    return False


def change_db_type(query_type: str, subsection_type: str, db_type: str):
    """
    Change the database to search.
    Args:
        query_type: The output type.
        db_type: The database to search.
    """
    if query_type == "text":
        subsection_update = gr.update(choices=list(valid_subsections[db_type]), value="Function")
    else:
        subsection_update = gr.update(visible=False)
    
    nprobe_visible = check_index_ivf(query_type, db_type, subsection_type)
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
                text_db = list(all_index["text"].keys())[0]
                sequence_db = list(all_index["sequence"].keys())[0]
                subsection_type = gr.Dropdown(valid_subsections[text_db], label="Subsection of text", value="Function",
                                              interactive=True, visible=False, scale=0)
                
                db_type = gr.Dropdown(all_index["sequence"].keys(), label="Database", value=sequence_db,
                                              interactive=True, visible=True, scale=0)

            with gr.Row():
                # Input box
                input = gr.Text(label="Input")
                
                # Provide an upload button to upload a pdb file
                upload_btn, chain_box = upload_pdb_button(visible=False, chain_visible=False)
                upload_btn.upload(parse_pdb_file, inputs=[input_type, upload_btn, chain_box], outputs=[input])
            
            
            # If the index is of IVF type, provide an option to choose the number of clusters.
            nprobe_visible = check_index_ivf(query_type.value, db_type.value)
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
            examples = gr.Dataset(samples=samples["text"], components=[input], label="Input examples")
            
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
            
        search_btn.click(fn=search, inputs=[input, nprobe, topk, input_type, query_type, subsection_type, db_type],
                      outputs=[results, download_btn, histogram])
        
        clear_btn.click(fn=clear_results, outputs=[results, download_btn, histogram])