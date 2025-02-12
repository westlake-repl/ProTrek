import gradio as gr
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import requests
import json
import time

from easydict import EasyDict
from scipy.stats import norm
# from .init_model import model, all_index, valid_subsections
from .blocks import upload_pdb_button, parse_pdb_file
from Bio.Align import PairwiseAligner
from utils.constants import sequence_level


tmp_file_path = "/tmp/results.tsv"
tmp_plot_path = "/tmp/histogram.png"
plot_available = True
record_available = True

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
        ["Catalyzes the hydrolysis of cutin, a polyester that forms the structure of plant cuticle "]
    ],
}

# Load all indexes and valid subsections
BASE_DIR = os.path.dirname(__file__)
config = EasyDict(yaml.safe_load(open(f"{BASE_DIR}/../config.yaml"))).frontend
all_index = {}

all_index["sequence"] = {}
for db in config.sequence:
    all_index["sequence"][db] = {}

all_index["structure"] = {}
for db in config.structure:
    all_index["structure"][db] = {}

# Load text index
all_index["text"] = {}
valid_subsections = {}
for db_name, subsections in config.text.items():
    all_index["text"][db_name] = {}

    valid_subsections[db_name] = set()
    for subsection in subsections:
        all_index["text"][db_name][subsection] = {}
        valid_subsections[db_name].add(subsection)

# Sort valid_subsections
for db_name in valid_subsections:
    valid_subsections[db_name] = sorted(list(valid_subsections[db_name]))


def clear_results():
    return "", gr.update(visible=False), gr.update(visible=False)


# Record visits
def record(record_dict: dict):
    global record_available
    while not plot_available:
        time.sleep(0.1)
    
    # Lock
    record_available = False
    
    # Add one to the number of visits
    cnt_file = f"{BASE_DIR}/../backend/statistics.tsv"
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    record_info = "\t".join(str(v) for v in record_dict.values())
    
    if not os.path.exists(cnt_file):
        with open(cnt_file, "w") as w:
            record_keys = "\t".join(list(record_dict.keys()))
            w.write(f"time\t{record_keys}\n")
    
    with open(cnt_file, "a") as w:
        w.write(f"{now_time}\t{record_info}\n")
        
    # Unlock
    record_available = True
    

def plot(scores) -> None:
    """
    Plot the distribution of scores and fit a normal distribution.
    Args:
        scores: List of scores
    """

    # Wait for the plot to be available
    global plot_available
    while not plot_available:
        time.sleep(0.1)

    # Lock the plot
    plot_available = False

    plt.hist(scores, bins=100, density=True, alpha=0.6)
    plt.title('Distribution of similarity scores in the database', fontsize=15)
    plt.xlabel('Similarity score', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    y_ed = plt.gca().get_ylim()[-1]
    plt.ylim(-0.05, y_ed)

    # Add note
    x_st = plt.gca().get_xlim()[0]
    text = ("Note: For the \"PDB\" and \"Swiss-Prot\" databases, the figure shows scores of\n "
            "all samples. For other databases, it displays the distribution of top-ranked\n "
            "clusters (identified using Faiss).")
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

    # Unlock the plot
    plot_available = True


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
    
    # Regulate the input
    if input_type == "sequence":
        input = input.upper()
        
    elif input_type == "structure":
        input = input.lower()
    
    input = input.replace("\n", "")
    
    # Send search request
    params = {
        "input": input,
        "topk": topk,
        "input_type": input_type,
        "query_type": query_type,
        "subsection_type": subsection_type,
        "db": db
    }
    
    try:
        url = f"http://127.0.0.1:7861/search"
        response = requests.get(url=url, params=params).json()
        with open(response["file_path"], "r") as r:
            response = json.load(r)
    except Exception as e:
        print(response)
        raise gr.Error("The system is busy. Please try again later.")
    
    # Record visits
    record(params)
    
    results = response["results"]
    all_scores = response["all_scores"]
    ids = response["ids"]

    plot(all_scores)
    
    # Set topk
    topk = len(results)

    # If both the input and output are protein sequences, calculate the sequence identity
    if input_type == "sequence" and query_type == "sequence":
        seq_identities = []
        for i in range(topk):
            hit_seq = ids[i].split("\t")[1]
            seq_identities.append(calc_seq_identity(input, hit_seq))

        seq_identities = [f"{identity * 100:.2f}%" for identity in seq_identities]

    # Write the results to a temporary file for downloading
    with open(tmp_file_path, "w") as w:
        seq_column_name = "Sequence" if query_type == "sequence" else "Foldseek sequence"
        
        if query_type == "text":
            w.write("Id\tMatching score\n")
            
        elif input_type == "sequence" and query_type == "sequence":
            w.write(f"Id\t{seq_column_name}\tLength\tSequence identity\tMatching score\n")
            
        else:
            w.write(f"Id\t{seq_column_name}\tLength\tMatching score\n")

        for i in range(topk):
            index_rk, score, rank = results[i]
            if query_type == "text":
                w.write(f"{ids[i]}\t{score}\n")

            elif input_type == "sequence" and query_type == "sequence":
                id, seq, length = ids[i].split("\t")
                w.write(f"{id}\t{seq}\t{length}\t{seq_identities[i]}\t{score}\n")

            else:
                id, seq, length = ids[i].split("\t")
                w.write(f"{id}\t{seq}\t{length}\t{score}\n")

    # Get topk ids
    topk_ids = []
    topk_scores = []
    topk_seqs = []
    topk_lengths = []
    for i in range(topk):
        index_rk, score, rank = results[i]
        now_id = ids[i].split("\t")[0].replace("|", "\\|")
        if query_type != "text":
            now_id = now_id[:20] + "..." if len(now_id) > 20 else now_id
        topk_scores.append(score)

        if query_type == "text":
            topk_ids.append(now_id)
        else:
            if db in ["UniRef50", "Uncharacterized", "Swiss-Prot", "Baker's yeast"]:
                # Provide link to uniprot website
                topk_ids.append(f"[{now_id}](https://www.uniprot.org/uniprotkb/{now_id})")
                
            elif db == "PDB":
                # Provide link to pdb website
                pdb_id = now_id.split("-")[0]
                topk_ids.append(f"[{now_id}](https://www.rcsb.org/structure/{pdb_id})")
                
            elif db == "NCBI":
                # Provide link to ncbi website
                topk_ids.append(f"[{now_id}](https://www.ncbi.nlm.nih.gov/protein/{now_id})")
                
            else:
                topk_ids.append(now_id)

        if query_type != "text":
            _, ori_seq, ori_len = ids[i].split("\t")
            seq = ori_seq[:20] + "..." if len(ori_seq) > 20 else ori_seq
            topk_seqs.append(seq)
            topk_lengths.append(ori_len)

    limit = 1000
    seq_column_name = "Sequence" if query_type == "sequence" else "Foldseek sequence"
    if query_type == "text":
        df = pd.DataFrame({"Id": topk_ids[:limit], "Matching score": topk_scores[:limit]})
        if len(topk_ids) > limit:
            info_df = pd.DataFrame({"Id": ["Download the file to check all results"], "Matching score": ["..."]},
                                   index=[1000])
            df = pd.concat([df, info_df], axis=0)

    elif input_type == "sequence" and query_type == "sequence":
        df = pd.DataFrame({"Id": topk_ids[:limit], seq_column_name: topk_seqs[:limit],
                           "Length": topk_lengths[:limit], "Sequence identity": seq_identities[:limit],
                          "Matching score": topk_scores[:limit]})
        if len(topk_ids) > limit:
            info_df = pd.DataFrame({"Id": ["Download the file to check all results"], seq_column_name: ["..."], "Length": ["..."],
                                    "Sequence identity": ["..."], "Matching score": ["..."]}, index=[1000])
            df = pd.concat([df, info_df], axis=0)

    else:
        df = pd.DataFrame({"Id": topk_ids[:limit], seq_column_name: topk_seqs[:limit], "Length": topk_lengths[:limit], "Matching score": topk_scores[:limit]})
        if len(topk_ids) > limit:
            info_df = pd.DataFrame({"Id": ["Download the file to check all results"], seq_column_name: ["..."], "Length": ["..."], "Matching score": ["..."]},
                                   index=[1000])
            df = pd.concat([df, info_df], axis=0)
    
    # This is to return the full results via Gradio API call
    hidden_df = pd.read_csv(tmp_file_path, sep="\t")
    
    output = df.to_markdown()
    return (
        output,
        gr.DownloadButton(label="Download results", value=tmp_file_path, visible=True, scale=0),
        gr.update(value=tmp_plot_path, visible=True),
        gr.DataFrame(hidden_df)
    )


def change_input_type(choice: str):
    # Change examples if input type is changed
    global samples
    
    # Set visibility of upload button
    if choice == "text":
        visible = False
        label = "Input (We recommend describing the protein's properties rather than using a simple numerical value like an EC number)"
        
    elif choice == "sequence":
        visible = True
        label = "Input (Paste a protein sequence. See examples below)"
    
    elif choice == "structure":
        visible = True
        label = "Input (Paste a protein 3Di sequence. See examples below)"
    
    return (
        gr.update(samples=samples[choice]),
        gr.update(label=label, value=""),
        gr.update(visible=visible),
        gr.update(visible=visible)
    )


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
    return False
    # if index_type == "sequence":
    #     index = all_index["sequence"][db]["index"]
    #
    # elif index_type == "structure":
    #     index = all_index["structure"][db]["index"]
    #
    # elif index_type == "text":
    #     index = all_index["text"][db][subsection_type]["index"]
    #
    # # nprobe_visible = True if hasattr(index, "nprobe") else False
    # # return nprobe_visible
    # return False


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
    gr.Markdown(f"# Search [protein databases](https://github.com/westlake-repl/ProTrek/wiki/Database-introduction) with [ProTrek](https://github.com/westlake-repl/ProTrek)")
    gr.Markdown(f"**Note: ProTrek does not support viral protein predictions for security reasons.**")
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
                    scale=1,
                )
                
                # If the output type is "text", provide an option to choose the subsection of text
                text_db = list(all_index["text"].keys())[0]
                sequence_db = list(all_index["sequence"].keys())[0]
                subsection_type = gr.Dropdown(valid_subsections[text_db], label="Subsection of text",
                                              value="Function",
                                              interactive=True, visible=False, scale=0)
                
                db_type = gr.Dropdown(all_index["sequence"].keys(), label="Database", value=sequence_db,
                                          interactive=True, visible=True, scale=0)
                
                # gr.Markdown("hello")
                
            with gr.Row():
                # Input box
                input = gr.Text(
                    label="Input (We recommend describing the protein's properties rather than using a simple numerical value like an EC number)"
                )
                
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
            topk = gr.Slider(1, 10000, 5,  step=1, label="Retrieve top k results")

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
                # This is to return the full results via Gradio API call
                hidden_results = gr.DataFrame(visible=False)
                download_btn = gr.DownloadButton(label="Download results", visible=False)
            
                # Plot the distribution of scores
                histogram = gr.Image(label="Histogram of matching scores", type="filepath", scale=1, visible=False)
            
        search_btn.click(fn=search, inputs=[input, nprobe, topk, input_type, query_type, subsection_type, db_type],
                      outputs=[results, download_btn, histogram, hidden_results], concurrency_limit=4)
        
        clear_btn.click(fn=clear_results, outputs=[results, download_btn, histogram])