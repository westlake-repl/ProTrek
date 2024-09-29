import sys
root_dir = __file__.rsplit("/", 2)[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)

import gradio as gr

from modules.search import build_search_module
from modules.compute_score import build_score_computation
from modules.tmalign import build_TMalign
from modules.cli import build_cli


# Build demo
with gr.Blocks() as demo:
    build_search_module()
    build_score_computation()
    build_TMalign()
    # build_cli()


if __name__ == '__main__':
    # Run the demo
    demo.launch()
