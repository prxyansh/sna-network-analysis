from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.utils import ImageReader

# Metadata
ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"
PDF_PATH = OUTDIR / "report_fillable.pdf"
DATA_PATH = ROOT / "email-Eu-core.txt"

STUDENT_NAME = "Priyansh Kumar Paswan"
ROLL_NO = "205124071"
SUBJECT = "Social Network Analysis"
PROFESSOR = "Dr. S.R. Balasundaram"
DEPARTMENT = "Computer Applications"
INSTITUTE = "National Institute of Technology, Tiruchirappalli"
LOGO_PATH = Path("/Users/prx./Documents/SNA/Project_finl/National_Institute_of_Technology,_Tiruchirappalli.svg.png")


def ensure_outputs_exist():
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


def styleset():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleLarge', parent=styles['Title'], fontSize=26, leading=32, alignment=1, spaceAfter=18)
    subtitle = ParagraphStyle(name='Subtitle', parent=styles['Heading2'], fontSize=14, leading=18, alignment=1, textColor=colors.grey)
    h2 = ParagraphStyle(name='H2', parent=styles['Heading2'], spaceBefore=12, spaceAfter=8)
    body = ParagraphStyle(name='Body', parent=styles['BodyText'], leading=14)
    small = ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=9, leading=12, textColor=colors.grey)
    return title_style, subtitle, h2, body, small


def add_image(flow: List, path: Path, caption: str, max_width: float = 6.0 * inch, max_height: float = 7.0 * inch):
    if path.exists():
        ir = ImageReader(str(path))
        iw, ih = ir.getSize()
        scale = min(max_width / float(iw), max_height / float(ih))
        w, h = iw * scale, ih * scale
        img = Image(str(path), width=w, height=h)
        flow.append(img)
        flow.append(Spacer(1, 0.1 * inch))
        styles = getSampleStyleSheet()
        flow.append(Paragraph(caption, styles['BodyText']))
        flow.append(Spacer(1, 0.15 * inch))


def add_blank_box(flow: List, label: str, height_in: float = 3.5):
    # Create a light shaded box of fixed height for user to paste screenshots or write notes
    width = A4[0] - 72  # page width minus margins (36 on each side)
    tbl = Table([[Paragraph(f"<b>{label}</b>", getSampleStyleSheet()['BodyText'])]], colWidths=[width], rowHeights=[height_in * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('BOX', (0,0), (-1,-1), 0.8, colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 0.15 * inch))


def build_pdf():
    ensure_outputs_exist()
    title_style, subtitle, h2, body, small = styleset()

    def _footer(canvas, doc):
        canvas.saveState()
        page = canvas.getPageNumber()
        footer_text = f"{SUBJECT} — {STUDENT_NAME} (Roll: {ROLL_NO}) — Page {page}"
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 36, 20, footer_text)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    flow: List = []

    # Cover
    flow.append(Spacer(1, 1.0 * inch))
    flow.append(Paragraph(SUBJECT, title_style))
    flow.append(Paragraph("Project Report (Fillable)", subtitle))
    flow.append(Spacer(1, 0.4 * inch))

    if LOGO_PATH.exists():
        ir = ImageReader(str(LOGO_PATH))
        iw, ih = ir.getSize()
        scale = min((2.6*inch)/float(iw), (2.0*inch)/float(ih))
        w, h = iw * scale, ih * scale
        logo_img = Image(str(LOGO_PATH), width=w, height=h)
        flow.append(logo_img)
        flow.append(Spacer(1, 0.15 * inch))
    flow.append(Paragraph(INSTITUTE, subtitle))
    flow.append(Spacer(1, 0.3 * inch))

    meta_tbl = Table([
        [Paragraph("<b>Student</b>", body), Paragraph(STUDENT_NAME, body)],
        [Paragraph("<b>Roll Number</b>", body), Paragraph(ROLL_NO, body)],
        [Paragraph("<b>Professor</b>", body), Paragraph(PROFESSOR, body)],
        [Paragraph("<b>Department</b>", body), Paragraph(DEPARTMENT, body)],
        [Paragraph("<b>Institute</b>", body), Paragraph(INSTITUTE, body)],
        [Paragraph("<b>Date</b>", body), Paragraph("________________________", body)],
        [Paragraph("<b>Signature</b>", body), Paragraph("________________________", body)],
    ], colWidths=[1.8*inch, 4.5*inch])
    meta_tbl.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#fbfbfb')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    flow.append(meta_tbl)
    flow.append(PageBreak())

    # 1. Overview
    flow.append(Paragraph("1. Dataset Overview", h2))
    flow.append(Paragraph("Email-EU-core: structural summary and degree characteristics.", body))

    G = nx.read_edgelist(DATA_PATH, create_using=nx.Graph(), nodetype=int)
    nodes = G.number_of_nodes(); edges = G.number_of_edges()
    density = nx.density(G); clustering = nx.average_clustering(G); is_conn = nx.is_connected(G)
    stats = [["Metric", "Value"],["Nodes", f"{nodes}"],["Edges", f"{edges}"],["Density", f"{density:.4f}"],["Avg clustering", f"{clustering:.4f}"],["Connected", f"{is_conn}"]]
    t = Table(stats, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 0.15 * inch))

    add_image(flow, OUTDIR / "subgraph.png", "Figure 1: Subgraph visualization")
    add_image(flow, OUTDIR / "degree_distribution.png", "Figure 2: Degree distribution histogram")
    add_blank_box(flow, "Paste Gephi screenshot: Global layout (ForceAtlas2), color=community, size=pagerank", height_in=4.0)
    add_blank_box(flow, "Your observations/notes for Section 1", height_in=1.5)
    flow.append(PageBreak())

    # 2. Link Analysis
    flow.append(Paragraph("2. Link Analysis (PageRank & Eigenvector)", h2))
    la = pd.read_csv(OUTDIR / "link_analysis.csv").sort_values("pagerank", ascending=False)
    data = [list(["node","pagerank","eigenvector","degree"])] + la[["node","pagerank","eigenvector","degree"]].head(10).values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),('GRID', (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    flow.append(table)
    add_image(flow, OUTDIR / "link_analysis_top_pagerank.png", "Figure 3: Top nodes by PageRank")
    add_blank_box(flow, "Paste Gephi screenshot: Influence view — size=pagerank, color=eigenvector gradient", height_in=3.5)
    add_blank_box(flow, "Your observations/notes for Section 2", height_in=1.5)
    flow.append(PageBreak())

    # 3. Node Classification
    flow.append(Paragraph("3. Node Classification (Label Propagation)", h2))
    labels = pd.read_csv(OUTDIR / "node_classification_labels.csv")
    comm_counts = labels.groupby("community").size().reset_index(name="size").sort_values("size", ascending=False)
    data = [list(["community","size"])] + comm_counts.head(15).values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),('GRID', (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    flow.append(table)
    add_blank_box(flow, "ADD GEPHI SCREENSHOT HERE: Community visualization — color by community, size by pagerank; layout: ForceAtlas2", height_in=4.0)
    add_blank_box(flow, "Your observations/notes for Section 3", height_in=1.5)
    flow.append(PageBreak())

    # 4. Influence Analysis
    flow.append(Paragraph("4. Influence Analysis (PageRank & Betweenness)", h2))
    inf = pd.read_csv(OUTDIR / "influence_scores.csv").sort_values(["pagerank","betweenness"], ascending=False)
    data = [list(["node","pagerank","betweenness","degree"])] + inf.head(10).values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),('GRID', (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    flow.append(table)
    add_image(flow, OUTDIR / "influence_top.png", "Figure 4: Top influencers")
    add_blank_box(flow, "ADD GEPHI SCREENSHOT HERE: Influencers — size by betweenness or pagerank; label top nodes", height_in=3.5)
    add_blank_box(flow, "Your observations/notes for Section 4", height_in=1.5)
    flow.append(PageBreak())

    # 5. Link Prediction
    flow.append(Paragraph("5. Link Prediction (Adamic-Adar, Jaccard, Preferential Attachment)", h2))
    auc = pd.read_csv(OUTDIR / "link_prediction_auc.csv").sort_values("auc", ascending=False)
    data = [list(["metric","auc"])] + auc.values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),('GRID', (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    flow.append(table)
    add_image(flow, OUTDIR / "link_prediction_roc.png", "Figure 5: ROC curves for link prediction")
    add_blank_box(flow, "Your observations/notes for Section 5", height_in=1.5)
    flow.append(PageBreak())

    # 6. Anomaly Detection
    flow.append(Paragraph("6. Anomaly Detection (IsolationForest)", h2))
    anom = pd.read_csv(OUTDIR / "anomalies.csv")
    top = anom.sort_values("decision_function").head(10)
    data = [list(["node","degree","clustering","avg_neighbor_degree","ego_edges","decision_function"])] + top.values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),('TEXTCOLOR', (0,0), (-1,0), colors.white),('GRID', (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    flow.append(table)
    add_image(flow, OUTDIR / "anomaly_scatter.png", "Figure 6: Anomaly scatter (Degree vs Clustering)")
    add_blank_box(flow, "ADD GEPHI SCREENSHOT HERE: Anomaly view — color by anomaly_score; highlight top outliers", height_in=3.5)
    add_blank_box(flow, "Your observations/notes for Section 6", height_in=1.5)
    flow.append(PageBreak())

    # 7. Gephi Visualizations
    flow.append(Paragraph("7. Gephi Visualizations (What to include)", h2))
    for b in [
        "Community view: Color by 'community' (categorical), size by 'pagerank'.",
        "Influence view: Size by 'betweenness' or 'pagerank'; optional color gradient by betweenness.",
        "Anomaly view: Color by 'anomaly_score' (continuous gradient).",
    ]:
        flow.append(Paragraph(f"• {b}", body))
    add_blank_box(flow, "Paste any additional screenshots here", height_in=4.0)

    doc.build(flow,
              onFirstPage=lambda c,d: None,
              onLaterPages=lambda c,d: None)
    print(f"✅ PDF created: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
