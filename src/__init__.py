import gradio as gr
from rag_pipeline import generate_answer

def ask_question(question):
    answer, sources = generate_answer(question)

    source_texts = "\n\n".join([
        f"Product: {s['metadata']['product']}\n{s['text'][:300]}..."
        for s in sources[:2]
    ])

    return answer, source_texts

with gr.Blocks() as demo:
    gr.Markdown("# 💬 CrediTrust Complaint Assistant")

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="AI Answer", lines=6)
    sources = gr.Textbox(label="Sources Used", lines=6)

    with gr.Row():
        submit = gr.Button("Ask")
        clear = gr.Button("Clear")

    submit.click(ask_question, inputs=question, outputs=[answer, sources])
    clear.click(lambda: ("", "", ""), outputs=[question, answer, sources])

demo.launch()
