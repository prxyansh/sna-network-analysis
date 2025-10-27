from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.utils import ImageReader
import networkx as nx

# Student/course metadata
STUDENT_NAME = "Priyansh Kumar Paswan"
ROLL_NO = "205124071"
SUBJECT = "Social Network Analysis"
PROFESSOR = "Dr. S.R. Balasundaram"
DEPARTMENT = "Computer Applications"
INSTITUTE = "National Institute of Technology, Tiruchirappalli"
LOGO_PATH = Path("/Users/prx./Documents/SNA/Project_finl/National_Institute_of_Technology,_Tiruchirappalli.svg.png")

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"
PDF_PATH = OUTDIR / "report.pdf"
DATA_PATH = ROOT / "email-Eu-core.txt"


def ensure_outputs_exist():
    # If key files missing, run project.py to regenerate
    needed = [
        OUTDIR / "subgraph.png",
        OUTDIR / "degree_distribution.png",
        OUTDIR / "link_analysis.csv",
        OUTDIR / "link_analysis_top_pagerank.png",
        OUTDIR / "influence_scores.csv",
        OUTDIR / "influence_top.png",
        OUTDIR / "node_classification_labels.csv",
        OUTDIR / "link_prediction_roc.png",
        OUTDIR / "link_prediction_auc.csv",
        OUTDIR / "anomaly_scatter.png",
        OUTDIR / "anomalies.csv",
        OUTDIR / "graph_attributes.gexf",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        import subprocess, sys
        print("Some outputs missing; running project.py to generate them...")
        cmd = [sys.executable, str(ROOT / "project.py"), "--task", "all", "--outdir", str(OUTDIR)]
        subprocess.check_call(cmd, cwd=ROOT)


def basic_stats_table() -> Table:
    G = nx.read_edgelist(DATA_PATH, create_using=nx.Graph(), nodetype=int)
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    is_conn = nx.is_connected(G)
    data = [["Metric", "Value"],
            ["Nodes", f"{nodes}"],
            ["Edges", f"{edges}"],
            ["Density", f"{density:.4f}"],
            ["Avg clustering", f"{clustering:.4f}"],
            ["Connected", f"{is_conn}"]]
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
    ]))
    return t


def df_table(df: pd.DataFrame, title: str = "", max_rows: int = 15, cols: List[str] | None = None) -> List:
    flow = []
    styles = getSampleStyleSheet()
    if title:
        flow.append(Paragraph(title, styles['Heading3']))
    if cols:
        df = df[cols]
    df = df.head(max_rows)
    data = [list(df.columns)] + df.values.tolist()
    tbl = Table(data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
    ]))
    flow.append(tbl)
    return flow


def add_image(flow: List, path: Path, caption: str, max_width: float = 6.0 * inch, max_height: float = 7.0 * inch):
    if path.exists():
        ir = ImageReader(str(path))
        iw, ih = ir.getSize()
        # scale to fit within max_width x max_height preserving aspect ratio
        scale = min(max_width / float(iw), max_height / float(ih))
        w = iw * scale
        h = ih * scale
        img = Image(str(path), width=w, height=h)
        flow.append(img)
        flow.append(Spacer(1, 0.15 * inch))
        styles = getSampleStyleSheet()
        flow.append(Paragraph(caption, styles['BodyText']))
        flow.append(Spacer(1, 0.2 * inch))


def build_pdf():
    ensure_outputs_exist()

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='TitleLarge', parent=styles['Title'], fontSize=26, leading=32, alignment=1, spaceAfter=18
    )
    subtitle = ParagraphStyle(
        name='Subtitle', parent=styles['Heading2'], fontSize=14, leading=18, alignment=1, textColor=colors.grey
    )
    h2 = ParagraphStyle(name='H2', parent=styles['Heading2'], spaceBefore=12, spaceAfter=8)
    h3 = ParagraphStyle(name='H3', parent=styles['Heading3'], spaceBefore=8, spaceAfter=6)
    body = ParagraphStyle(name='Body', parent=styles['BodyText'], leading=14)
    small = ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=9, leading=12, textColor=colors.grey)

    # Footer with page numbers
    def _footer(canvas, doc):
        canvas.saveState()
        page = canvas.getPageNumber()
        footer_text = f"{SUBJECT} — {STUDENT_NAME} (Roll: {ROLL_NO}) — Page {page}"
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 36, 20, footer_text)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    flow: List = []

    # Helper: neat placeholder box for Gephi screenshots or manual edits
    def add_placeholder_box(text: str):
        tbl = Table([[Paragraph(f"<b>** {text} **</b>", body)]], colWidths=[A4[0] - 72])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
            ('BOX', (0,0), (-1,-1), 0.8, colors.lightgrey),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        flow.append(tbl)
        flow.append(Spacer(1, 0.15 * inch))

    # Cover Page
    flow.append(Spacer(1, 1.0 * inch))
    flow.append(Paragraph(SUBJECT, title_style))
    flow.append(Paragraph("Project Report", subtitle))
    flow.append(Spacer(1, 0.6 * inch))

    # Institute logo (centered) and name
    if LOGO_PATH.exists():
        ir = ImageReader(str(LOGO_PATH))
        iw, ih = ir.getSize()
        max_w, max_h = 2.6 * inch, 2.0 * inch
        scale = min(max_w / float(iw), max_h / float(ih))
        w, h = iw * scale, ih * scale
        logo_img = Image(str(LOGO_PATH), width=w, height=h)
        logo_img.hAlign = 'CENTER'
        flow.append(logo_img)
        flow.append(Spacer(1, 0.15 * inch))
    flow.append(Paragraph(INSTITUTE, subtitle))
    flow.append(Spacer(1, 0.4 * inch))

    meta_tbl = Table([
        [Paragraph("<b>Student</b>", body), Paragraph(STUDENT_NAME, body)],
        [Paragraph("<b>Roll Number</b>", body), Paragraph(ROLL_NO, body)],
        [Paragraph("<b>Professor</b>", body), Paragraph(PROFESSOR, body)],
        [Paragraph("<b>Department</b>", body), Paragraph(DEPARTMENT, body)],
        [Paragraph("<b>Institute</b>", body), Paragraph(INSTITUTE, body)],
    ], colWidths=[1.8*inch, 4.5*inch])
    meta_tbl.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f7f7f7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fbfbfb')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    flow.append(meta_tbl)
    flow.append(Spacer(1, 0.4 * inch))
    flow.append(Paragraph("Tools used: Python (NetworkX, pandas, scikit‑learn, matplotlib), Gephi.", small))
    flow.append(PageBreak())

    # Overview
    flow.append(Paragraph("1. Dataset Overview", h2))
    flow.append(Paragraph("Email-EU-core undirected graph. The analysis below summarizes structure and degree characteristics.", body))
    flow.append(Spacer(1, 0.1 * inch))
    flow.append(basic_stats_table())
    flow.append(Spacer(1, 0.2 * inch))
    add_image(flow, OUTDIR / "subgraph.png", "Figure 1: Subgraph visualization")
    add_image(flow, OUTDIR / "degree_distribution.png", "Figure 2: Degree distribution histogram")
    add_placeholder_box("Optional Gephi screenshot: Global layout (ForceAtlas2), color by community (from node_classification), size by pagerank")
    flow.append(PageBreak())

    # Link Analysis
    flow.append(Paragraph("2. Link Analysis (PageRank & Eigenvector)", h2))
    flow.append(Paragraph("Methods: PageRank for importance via random walks; Eigenvector centrality for importance via influential neighbors.", body))
    la = pd.read_csv(OUTDIR / "link_analysis.csv").sort_values("pagerank", ascending=False)
    flow += df_table(la, title="Top-10 by PageRank", max_rows=10, cols=["node", "pagerank", "eigenvector", "degree"])
    add_image(flow, OUTDIR / "link_analysis_top_pagerank.png", "Figure 3: Top nodes by PageRank (bar chart)")
    add_placeholder_box("Optional Gephi screenshot: Influence view — size by pagerank, color gradient by eigenvector")
    flow.append(PageBreak())

    # Node Classification
    flow.append(Paragraph("3. Node Classification (Label Propagation)", h2))
    flow.append(Paragraph("Method: Asynchronous label propagation to discover communities without ground truth labels.", body))
    labels = pd.read_csv(OUTDIR / "node_classification_labels.csv")
    comm_counts = labels.groupby("community").size().reset_index(name="size").sort_values("size", ascending=False)
    flow += df_table(comm_counts, title="Communities by size (Top 15)", max_rows=15, cols=["community", "size"])
    add_placeholder_box("ADD GEPHI SCREENSHOT HERE: Community visualization — color by community (categorical), size by pagerank; layout: ForceAtlas2")
    flow.append(PageBreak())

    # Influence Analysis
    flow.append(Paragraph("4. Influence Analysis (PageRank & Betweenness)", h2))
    flow.append(Paragraph("Methods: PageRank for global influence; Betweenness for brokerage across communities.", body))
    inf = pd.read_csv(OUTDIR / "influence_scores.csv").sort_values(["pagerank", "betweenness"], ascending=False)
    flow += df_table(inf, title="Top-10 influencers", max_rows=10, cols=["node", "pagerank", "betweenness", "degree"])
    add_image(flow, OUTDIR / "influence_top.png", "Figure 4: Top influencers (PageRank + Betweenness)")
    add_placeholder_box("ADD GEPHI SCREENSHOT HERE: Influencers — size by betweenness or pagerank; optionally label top nodes")
    flow.append(PageBreak())

    # Link Prediction
    flow.append(Paragraph("5. Link Prediction (Adamic-Adar, Jaccard, Preferential Attachment)", h2))
    flow.append(Paragraph("Procedure: Hold out 10% edges; score with three heuristics on the train graph; evaluate ROC/AUC.", body))
    auc = pd.read_csv(OUTDIR / "link_prediction_auc.csv").sort_values("auc", ascending=False)
    flow += df_table(auc, title="AUC by metric", max_rows=5, cols=["metric", "auc"])
    add_image(flow, OUTDIR / "link_prediction_roc.png", "Figure 5: ROC curves for link prediction metrics")
    flow.append(PageBreak())

    # Anomaly Detection
    flow.append(Paragraph("6. Anomaly Detection (IsolationForest on egonet features)", h2))
    flow.append(Paragraph("Features: degree, clustering, average neighbor degree, egonet edges; Model: IsolationForest (1% contamination).", body))
    anom = pd.read_csv(OUTDIR / "anomalies.csv")
    # Lower decision_function => more anomalous; show 10 lowest
    anom_top = anom.sort_values("decision_function").head(10)
    flow += df_table(anom_top, title="Top-10 anomalous nodes", max_rows=10, cols=["node", "degree", "clustering", "avg_neighbor_degree", "ego_edges", "decision_function"])
    add_image(flow, OUTDIR / "anomaly_scatter.png", "Figure 6: Anomaly scatter (Degree vs Clustering, colored by score)")
    add_placeholder_box("ADD GEPHI SCREENSHOT HERE: Anomaly view — color by anomaly_score (continuous), highlight top outliers")
    flow.append(PageBreak())

    # Gephi Section
    flow.append(Paragraph("7. Gephi Visualizations (What to include)", h2))
    flow.append(Paragraph("Open outputs/graph_attributes.gexf in Gephi and capture these views:", body))
    bullets = [
        "Community view: Color by 'community' (categorical), size by 'pagerank'.",
        "Influence view: Size by 'betweenness' or 'pagerank'; optional color gradient by betweenness.",
        "Anomaly view: Color by 'anomaly_score' (continuous gradient).",
    ]
    for b in bullets:
        flow.append(Paragraph(f"• {b}", body))
    flow.append(Spacer(1, 0.2 * inch))

    # Build with footer on all pages
    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    print(f"✅ PDF created: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
