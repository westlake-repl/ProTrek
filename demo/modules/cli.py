import gradio as gr
import subprocess


def run_command(cmd: str) -> str:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()

    if stdout:
        return f"[Output]\n{stdout.decode()}"
    if stderr:
        return f"[Error]\n{stderr.decode()}"


# Build the block for command line interface
def build_cli():
    gr.Markdown(f"# Input your command and click to run")
    with gr.Column():
        cmd = gr.Textbox(label="Input your command", value="echo 'Hello, World!'")
        btn = gr.Button(value="Run")
        output = gr.TextArea(label="Output", interactive=False)

        btn.click(run_command, inputs=[cmd], outputs=[output])