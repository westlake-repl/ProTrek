import gradio as gr
import torch

from .init_model import model
from utils.foldseek_util import get_struc_seq


input_types = ["protein sequence", "protein structure", "text"]

input_examples = {
    "protein sequence": [
        "MQLQRLGAPLLKRLVGGCIRQSTAPIMPCVVVSGSGGFLTPVRTYMPLPNDQSDFSPYIEIDLPSESRIQSLHKSGLAAQEWVACEKVHGTNFGIYLINQGDHEVVRFAKRSGIMDPNENFFGYHILIDEFTAQIRILNDLLKQKYGLSRVGRLVLNGELFGAKYKHPLVPKSEKWCTLPNGKKFPIAGVQIQREPFPQYSPELHFFAFDIKYSVSGAEEDFVLLGYDEFVEFSSKVPNLLYARALVRGTLDECLAFDVENFMTPLPALLGLGNYPLEGNLAEGVVIRHVRRGDPAVEKHNVSTIIKLRCSSFMELKHPGKQKELKETFIDTVRSGALRRVRGNVTVISDSMLPQVEAAANDLLLNNVSDGRLSNVLSKIGREPLLSGEVSQVDVALMLAKDALKDFLKEVDSLVLNTTLAFRKLLITNVYFESKRLVEQKWKELMQEEAAAQSEAIPPLSPAAPTKGE",
        "MSLSTEQMLRDYPRSMQINGQIPKNAIHETYGNDGVDVFIAGSGPIGATYAKLCVEAGLRVVMVEIGAADSFYAVNAEEGTAVPYVPGYHKKNEIEFQKDIDRFVNVIKGALQQVSVPVRNQNVPTLDPGAWSAPPGSSAISNGKNPHQREFENLSAEAVTRGVGGMSTHWTCSTPRIHPPMESLPGIGRPKLSNDPAEDDKEWNELYSEAERLIGTSTKEFDESIRHTLVLRSLQDAYKDRQRIFRPLPLACHRLKNAPEYVEWHSAENLFHSIYNDDKQKKLFTLLTNHRCTRLALTGGYEKKIGAAEVRNLLATRNPSSQLDSYIMAKVYVLASGAIGNPQILYNSGFSGLQVTPRNDSLIPNLGRYITEQPMAFCQIVLRQEFVDSVRDDPYGLPWWKEAVAQHIAKNPTDALPIPFRDPEPQVTTPFTEEHPWHTQIHRDAFSYGAVGPEVDSRVIVDLRWFGATDPEANNLLVFQNDVQDGYSMPQPTFRYRPSTASNVRARKMMADMCEVASNLGGYLPTSPPQFMDPGLALHLAGTTRIGFDKATTVADNNSLVWDFANLYVAGNGTIRTGFGENPTLTSMCHAIKSARSIINTLKGGTDGKNTGEHRNL",
        "MGVHECPAWLWLLLSLLSLPLGLPVLGAPPRLICDSRVLERYLLEAKEAENITTGCAEHCSLNENITVPDTKVNFYAWKRMEVGQQAVEVWQGLALLSEAVLRGQALLVNSSQPWEPLQLHVDKAVSGLRSLTTLLRALGAQKEAISPPDAASAAPLRTITADTFRKLFRVYSNFLRGKLKLYTGEACRTGDR"
    ],
    
    "protein structure": [
        "ddddddddddddddddddddddddddddddddpdpddpddpqpdddfddpdqqlddadddfaaddpvqvvlcvvvvvlqakkfkwfdadffkkkwkwadpdpdidifidtnvgtdglqpddllclvcvvlsvqlvvllqvvvcvvvvapafrmkmfiwgkdalddpfppadadpdwhagsvgdidgsvpgdrdddpaqhahsdiaietewiwiarnsdpvriqtafqvvvcvsqvprpphhyidgqfmggnllnlldpqqpaaqlrnqqvvnqvgddpprggqfikmfrrpprppvvcvsvrhgihtdghlvnvcvvdppcsvvcccnrcvprnvvscvvvvndhdtdvlsrhhpvlsvllvqllvlldpvllvvldvvvdlpclqvvvqdllnsllsslvvsvvvsvvpddpvnvpgdpvsvvvssvsssvsssvvsvvcvvvvnvvsvvvvvvvddppdpdddpddd",
        "dpdplvvqppdddplqappppfaadpvcvlvdpvaaaeeeeaqallsllllllclvlvgfyeyefqaeqpdwdddpddvpdddftqtqfapcqppvclqpqqvllvvqvvfwdwqeaefdqpppvpddppddhddppdgdddqqhdppfdpqqdlgqatwgghrntcqnhdpqfddawadadpvahqgtfdaldpdpvvrvvlvvvllvvlcvqlvkdqclqvpflqqcllqvllcvvcvvppwhkgggtgswhadpvhsldirhttsssscvvqrvdpssvssydyhyskhqqewhaghdpfgetawtkiarnccvvpvpdrgihigghrfyeypralprvllrcvssvqalqdpggdprhnqdqffalkwfwwkkkfkfffdpvsqvcqcvppppdpssnvqlvvqcvvcvpdpgsgdssrakhfmwtdadpvqqktktwidghhndddddppddpsrmimimiihwafrdrqfgwgfdppgdhpvrttrihtrddgdpvsvvsvvvrlvvsvvssvstgdtdprgpididrrnsvnlieqrqaedddsvngqayqlqhgpsyphygyfdrnhrngigngdcvsvrssssvsnsvvsscvvvvdpdddppdddddd",
        "ddppppdcvvvvvvvvvppppppvppldplvvlldvvllvvqlvllvvllvvcvvpdpnfflqdwqkafdlddpvvvvvpddlllllqlllvrlvsllvrlvsslvslvpdpdrdvvnnvssvvlnvssvvvnvssvslvsvvsnppddppprdddgdididrgssvssvsvssnsvgsvvvssvvssvvvvd"
    ],

    "text": [
        "RNA-editing ligase in kinetoplastid mitochondrial.",
        "Oxidase which catalyzes the oxidation of various aldopyranoses and disaccharides.",
        "Erythropoietin for regulation of erythrocyte proliferation and differentiation."
    ]
}

samples = [[s1, s2] for s1, s2 in zip(input_examples["protein sequence"], input_examples["text"])]


def compute_score(input_type_1: str, input_1: str, input_type_2: str, input_2: str):
    with torch.no_grad():
        input_reprs = []
        
        for input_type, input in [(input_type_1, input_1), (input_type_2, input_2)]:
            if input_type == "protein sequence":
                input_reprs.append(model.get_protein_repr([input]))
            
            elif input_type == "protein structure":
                input_reprs.append(model.get_structure_repr([input]))
            
            else:
                input_reprs.append(model.get_text_repr([input]))
        
        score = input_reprs[0] @ input_reprs[1].T / model.temperature
    
    return f"{score.item():.4f}"


# Convert pdb file to aa sequence or foldseek sequence
def parse_pdb(file, input_type):
    parsed_seqs = get_struc_seq("bin/foldseek", file)
    for seqs in parsed_seqs.values():
        if input_type == "protein sequence":
            return seqs[0]
        else:
            return seqs[1].lower()


def change_input_type(choice_1: str, choice_2: str):
    examples_1 = input_examples[choice_1]
    examples_2 = input_examples[choice_2]
    
    # Change examples if input type is changed
    global samples
    samples = [[s1, s2] for s1, s2 in zip(examples_1, examples_2)]
    
    # Set visibility of upload button
    if choice_1 == "text":
        visible_1 = False
    else:
        visible_1 = True
    
    if choice_2 == "text":
        visible_2 = False
    else:
        visible_2 = True
    
    return samples, "", "", gr.update(visible=visible_1), gr.update(visible=visible_2)


# Load example from dataset
def load_example(example_id):
    return samples[example_id]


# Build the block for computing protein-text similarity
def build_score_computation():
    gr.Markdown(f"# Compute similarity score between two modalities")
    with gr.Row(equal_height=True):
        with gr.Column():
            # Compute similarity score between sequence and text
            with gr.Row():
                input_1 = gr.Textbox(label="Input 1")
                
                # Choose the type of input 1
                input_type_1 = gr.Dropdown(input_types, label="Input type", value="protein sequence",
                                             interactive=True, visible=True)
                
                # Provide an upload button to upload a pdb file
                upload_btn_1 = gr.UploadButton(label="Upload .pdb/.cif file", scale=0)
                upload_btn_1.upload(parse_pdb, inputs=[upload_btn_1, input_type_1], outputs=[input_1])
            
            with gr.Row():
                input_2 = gr.Textbox(label="Input 2")
                
                # Choose the type of input 2
                input_type_2 = gr.Dropdown(input_types, label="Input type", value="text",
                                           interactive=True, visible=True)
                
                # Provide an upload button to upload a pdb file
                upload_btn_2 = gr.UploadButton(label="Upload .pdb/.cif file", scale=0, visible=False)
                upload_btn_2.upload(parse_pdb, inputs=[upload_btn_2, input_type_2], outputs=[input_2])
            
            # Provide examples
            examples = gr.Dataset(samples=samples, type="index", components=[input_1, input_2], label="Input examples")
            
            # Add click event to examples
            examples.click(fn=load_example, inputs=[examples], outputs=[input_1, input_2])
            
            compute_btn = gr.Button(value="Compute")
            
        # Change examples based on input type
        input_type_1.change(fn=change_input_type, inputs=[input_type_1, input_type_2],
                            outputs=[examples, input_1, input_2, upload_btn_1, upload_btn_2])
        
        input_type_2.change(fn=change_input_type, inputs=[input_type_1, input_type_2],
                            outputs=[examples, input_1, input_2, upload_btn_1, upload_btn_2])

        similarity_score = gr.Label(label="similarity score")
        compute_btn.click(fn=compute_score, inputs=[input_type_1, input_1, input_type_2, input_2],
                          outputs=[similarity_score])