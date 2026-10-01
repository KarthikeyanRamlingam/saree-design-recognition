from __future__ import annotations

import argparse

import gradio as gr
from PIL import Image

from src.inference import SareeSearch


def create_app(engine: SareeSearch, threshold: float):
    def search(image: Image.Image):
        if image is None:
            return [], []
        results = engine.search(image)
        gallery = [(row["image_path"], f"{row['design_id']} | {row['similarity']:.3f}") for row in results]
        table = [[row["design_id"], round(row["similarity"], 4), row["image_path"]] for row in results]
        return gallery, table

    def verify(first: Image.Image, second: Image.Image):
        if first is None or second is None:
            return "Upload both images."
        result = engine.verify(first, second, threshold)
        decision = "SAME DESIGN" if result["same_design"] else "DIFFERENT DESIGN"
        return f"{decision} — similarity {result['similarity']:.4f} (threshold {threshold:.4f})"

    with gr.Blocks(title="Saree Design Recognition") as demo:
        gr.Markdown("# Color-Invariant Saree Design Recognition")
        with gr.Tab("Identify"):
            query = gr.Image(type="pil", label="Query saree")
            button = gr.Button("Find matching designs", variant="primary")
            gallery = gr.Gallery(label="Top matches", columns=5)
            table = gr.Dataframe(headers=["Design", "Similarity", "Path"], interactive=False)
            button.click(search, query, [gallery, table])
        with gr.Tab("Verify"):
            with gr.Row():
                first = gr.Image(type="pil", label="First saree")
                second = gr.Image(type="pil", label="Second saree")
            verify_button = gr.Button("Compare", variant="primary")
            verdict = gr.Textbox(label="Result")
            verify_button.click(verify, [first, second], verdict)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery-csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    engine = SareeSearch(args.checkpoint, args.gallery_csv)
    create_app(engine, args.threshold).launch(share=args.share)


if __name__ == "__main__":
    main()
