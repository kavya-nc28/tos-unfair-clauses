"""
Simple web UI for uploading a ToS and viewing highlighted unfair clauses.

TODO:
- Implement Gradio (or similar) interface.
- Use pdf_to_text / split_into_clauses + trained model + severity mapping.
"""

import gradio as gr

from src.data.utils_pdf_text import pdf_to_text, split_into_clauses
from src.frontend.severity_mapping import logits_to_severity, severity_to_label

# TODO: load tokenizer + trained model checkpoint here.


def analyze_tos(file_or_text):
    """
    Main function called by the UI.

    Input can be:
    - uploaded PDF file
    - pasted raw text
    """
    # TODO:
    # 1. If file is PDF, extract text; else assume raw text.
    # 2. Split into clauses.
    # 3. Run model on each clause.
    # 4. Compute severity + label per clause.
    # 5. Return something like a list of (clause, severity, label).
    raise NotImplementedError("Implement analyze_tos().")


def build_interface():
    """
    Define Gradio interface layout.
    """
    # TODO: design minimal UI: file upload, text area, results area.
    raise NotImplementedError("Implement Gradio interface.")


if __name__ == "__main__":
    # TODO: call build_interface().launch()
    pass
