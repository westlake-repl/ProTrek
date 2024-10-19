import sys
root_dir = __file__.rsplit("/", 2)[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)

import gradio as gr
import os

from modules.search import build_search_module
from modules.compute_score import build_score_computation
from modules.tmalign import build_TMalign
from modules.cli import build_cli
from gradio.themes import *


# Build demo
with gr.Blocks(title="ProTrek") as demo:
    build_search_module()
    build_score_computation()
    build_TMalign()
    # build_cli()
 

if __name__ == '__main__':
    # Run the demo
    demo.launch(server_name="0.0.0.0")
