from __future__ import annotations
import gradio as gr
import requests
import math

from src.data.utils_pdf_text import pdf_to_text
from src.inference.preprocess_input import load_text_input


# =========================
# PREPROCESS
# =========================
def handle_upload(file):
    if file is None:
        return "❌ No file", []

    try:
        file_path = file.name if hasattr(file, "name") else file
        if file_path.endswith(".pdf"):
            raw = pdf_to_text(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()

        clauses = load_text_input(raw)

        return f"✅ Ready ({len(clauses)} clauses)", clauses

    except Exception as e:
        return f"❌ Error: {str(e)}", []


def handle_paste(text):
    if not text:
        return "❌ No text", []

    clauses = load_text_input(text)

    return f"✅ Ready ({len(clauses)} clauses)", clauses

# =========================
# GAUGE
# =========================
def build_gauge(safety_score: int) -> str:
    # ADD: semi-circular gauge for overall safety
    angle     = safety_score / 100 * 180           # 0 = fully unsafe, 180 = fully safe
    rad       = math.radians(angle)
    needle_x  = round(100 - 75 * math.cos(rad), 1)
    needle_y  = round(100 - 75 * math.sin(rad), 1)
    arc_dash  = round(safety_score * 2.83)         # 100% = 283 (π * 90)

    if safety_score >= 75:
        arc_color, verdict = "#10b981", "✅ You are good to go"
    elif safety_score >= 50:
        arc_color, verdict = "#f59e0b", "🟡 Needs another look"
    elif safety_score >= 25:
        arc_color, verdict = "#ef4444", "🟠 This might be trouble"
    else:
        arc_color, verdict = "#7c3aed", "🔴 Do NOT agree to this"

    return f"""
    <div style="text-align:center;margin:24px 0">
      <svg viewBox="0 0 200 115" width="280">
        <path d="M 10 100 A 90 90 0 0 1 190 100"
              fill="none" stroke="#1e293b" stroke-width="18" stroke-linecap="round"/>
        <path d="M 10 100 A 90 90 0 0 1 190 100"
              fill="none" stroke="{arc_color}" stroke-width="18" stroke-linecap="round"
              stroke-dasharray="{arc_dash} 283"/>
        <line x1="100" y1="100" x2="{needle_x}" y2="{needle_y}"
              stroke="white" stroke-width="3" stroke-linecap="round"/>
        <circle cx="100" cy="100" r="5" fill="white"/>
        <text x="100" y="86" text-anchor="middle"
              font-size="22" font-weight="bold" fill="white">{safety_score}%</text>
        <text x="100" y="110" text-anchor="middle"
              font-size="11" fill="#94a3b8">Overall Safety Score</text>
      </svg>
      <p style="color:{arc_color};font-size:16px;font-weight:bold;margin-top:4px">
        {verdict}
      </p>
    </div>
    """

# =========================
# BUILD CARDS
# =========================
def build_cards(results, filter_value):

    if filter_value != "ALL":
        results = [r for r in results if r["severity_band"] == filter_value]

    if not results:
        return "<p style='color:#94a3b8;padding:20px'>No clauses match this filter.</p>"
    
    color_map = {
        "CRITICAL": "#7c3aed",
        "HIGH":     "#ef4444",
        "MEDIUM":   "#f59e0b",
        "SAFE":     "#10b981",
    }

    cards_html = ""
    
    for r in results:
        color = color_map.get(r["severity_band"], "#94a3b8")

        verdict_line = (
            f'<p style="font-size:13px;color:{color};font-weight:bold;margin-top:6px">'
            f'⚠️ {r["verdict"]}</p>'
            if r.get("verdict") else ""
        )

        cards_html += f"""
        <div style="background:#0f172a;color:white;padding:20px;border-radius:12px;
                    margin-top:15px;border-left:6px solid {color}">
            <span style="background:{color};padding:5px 10px;
                         border-radius:12px;font-size:12px">
                {r['severity_band']}
            </span>
            <p style="margin-top:10px">{r['text']}</p>
            <p style="font-size:13px;color:#cbd5e1">💡 {r.get('explanation', '')}</p>
            {verdict_line}
            <p style="font-size:12px;color:#94a3b8">
                Score: {r.get('severity_score', '')}/10
            </p>
        </div>
        """
    return cards_html


# =========================
# CALL API
# =========================
def call_api(clauses):
    if not clauses:
        return "<p style='color:red'>No data to analyze.</p>", "", "", []
    
    try:
        res = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"clauses": clauses},
            timeout=120
        )
        res.raise_for_status()

        data = res.json()
        results = data.get("results", [])

        if not results:
            return "<p style='color:red'>No results returned.</p>", "", "", []

        # Counts
        critical: int = sum(1 for r in results if r["severity_band"] == "CRITICAL")
        high:     int = sum(1 for r in results if r["severity_band"] == "HIGH")
        medium:   int = sum(1 for r in results if r["severity_band"] == "MEDIUM")
        safe:     int = sum(1 for r in results if r["severity_band"] == "SAFE")

        summary_html = f"""
        <div style="display:flex;gap:20px;margin-top:20px;flex-wrap:wrap">
            <div style="flex:1;min-width:100px;background:#7c3aed;color:white;
                        padding:20px;border-radius:12px;text-align:center">
                <h1>{critical}</h1><p>Critical</p>
            </div>
            <div style="flex:1;min-width:100px;background:#ef4444;color:white;
                        padding:20px;border-radius:12px;text-align:center">
                <h1>{high}</h1><p>High Risk</p>
            </div>
            <div style="flex:1;min-width:100px;background:#f59e0b;color:white;
                        padding:20px;border-radius:12px;text-align:center">
                <h1>{medium}</h1><p>Medium</p>
            </div>
            <div style="flex:1;min-width:100px;background:#10b981;color:white;
                        padding:20px;border-radius:12px;text-align:center">
                <h1>{safe}</h1><p>Safe</p>
            </div>
        </div>
        """

        safety_score = int(data.get("safety_score", 50))
        gauge_html   = build_gauge(safety_score)

        cards_html = build_cards(results, "ALL")
        return summary_html, gauge_html, cards_html, results

    except Exception as e:
        return f"<p style='color:red'>API Error: {str(e)}</p>", "", "", []


# =========================
# FILTER
# =========================
def apply_filter(filter_value, results):
    if not results:
        return ""

    return build_cards(results, filter_value)


# =========================
# UI
# =========================
with gr.Blocks(title="ToS Risk Analyzer") as demo:

    gr.Markdown("## 🚀 ToS Risk Analyzer")

    state = gr.State([])
    results_state = gr.State([])

    with gr.Row():
        file = gr.File(label="Upload ToS (.txt / .pdf)")
        text = gr.Textbox(label="Paste Text", lines=10)

    status = gr.Textbox(label="Status")

    load_btn = gr.Button("📄 Load Text", variant="secondary")
    load_btn.click(handle_paste, inputs=text, outputs=[status, state])

    file.change(handle_upload, inputs=file, outputs=[status, state])

    analyze_btn = gr.Button("✨ Analyze")

    summary = gr.HTML()
    gauge = gr.HTML()

    filter_radio = gr.Radio(
        choices=["ALL", "CRITICAL", "HIGH", "MEDIUM", "SAFE"],
        value="ALL",
        label="Filter Results"
    )

    cards = gr.HTML()

    analyze_btn.click(
        fn=call_api,
        inputs=[state],
        outputs=[summary, gauge, cards, results_state],  
        show_progress="minimal",           
    )
    filter_radio.change(
        fn=apply_filter,
        inputs=[filter_radio, results_state],
        outputs=[cards],
    )


demo.launch()