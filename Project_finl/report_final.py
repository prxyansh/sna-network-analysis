from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import random

import pandas as pd
import networkx as nx

# Use headless backend for plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"
DATA_PATH = ROOT / "email-Eu-core.txt"
PDF_PATH = OUTDIR / "report_final.pdf"

# Metadata
STUDENT_NAME = "Priyansh Kumar Paswan"
ROLL_NO = "205124071"
SUBJECT = "Social Network Analysis"
PROFESSOR = "Dr. S.R. Balasundaram"
DEPARTMENT = "Computer Applications"
INSTITUTE = "National Institute of Technology, Tiruchirappalli"
LOGO_PATH = Path("/Users/prx./Documents/SNA/Project_finl/National_Institute_of_Technology,_Tiruchirappalli.svg.png")


NEEDED_OUTPUTS = [
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
]

# Paths for Python-generated 'Gephi-like' images
COMMUNITY_IMG = OUTDIR / "community_view.png"
INFLUENCE_IMG = OUTDIR / "influence_view.png"
ANOMALY_IMG = OUTDIR / "anomaly_view.png"
LINK_ANALYSIS_IMG = OUTDIR / "link_analysis_view.png"


def ensure_outputs_exist():
    missing = [p for p in NEEDED_OUTPUTS if not p.exists()]
    if missing:
        import subprocess, sys
        print("Some outputs missing; running project.py to generate them...")
        cmd = [sys.executable, str(ROOT / "project.py"), "--task", "all", "--outdir", str(OUTDIR)]
        subprocess.check_call(cmd, cwd=ROOT)


def _scale(values, min_size=30, max_size=300):
    import numpy as np
    v = pd.Series(values, dtype=float)
    if (v.max() - v.min()) < 1e-12:
        return [min_size for _ in v]
    s = (v - v.min()) / (v.max() - v.min())
    return list(min_size + s * (max_size - min_size))


def _positions(G: nx.Graph) -> Dict[int, tuple]:
    # Spring layout with fixed seed; modest iterations for speed
    return nx.spring_layout(G, seed=42, iterations=50)


def _draw_edges_sampled(G: nx.Graph, pos: Dict[int, tuple], sample=8000, alpha=0.05, linewidth=0.4):
    edges = list(G.edges())
    if len(edges) > sample:
        random.seed(42)
        edges = random.sample(edges, sample)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=linewidth, alpha=alpha, edge_color="#888888")


def generate_community_view(G: nx.Graph, labels_df: pd.DataFrame, pr_df: pd.DataFrame, out_path: Path):
    pos = _positions(G)
    # Map node -> community and pagerank
    node2comm = dict(zip(labels_df["node"], labels_df["community"]))
    node2pr = dict(zip(pr_df["node"], pr_df["pagerank"]))
    # Colors by community using tab20
    comms = pd.Series(node2comm)
    comm_ids = sorted(comms.unique())
    base_cmap = matplotlib.colormaps.get_cmap("tab20")
    # Sample evenly from the base colormap for distinct community colors
    if len(comm_ids) <= 1:
        color_list = [base_cmap(0.0)]
    else:
        color_list = [base_cmap(i / max(1, len(comm_ids)-1)) for i in range(len(comm_ids))]
    comm2color = {cid: color_list[i % len(color_list)] for i, cid in enumerate(comm_ids)}
    node_colors = [comm2color.get(node2comm.get(n, comm_ids[0]), (0.6,0.6,0.6,1.0)) for n in G.nodes()]
    node_sizes = _scale([node2pr.get(n, 0.0) for n in G.nodes()], min_size=20, max_size=220)

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_edges_sampled(G, pos)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0.2, edgecolors="#ffffff")
    ax.set_title("Community View (color=community, size=pagerank)")
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_influence_view(G: nx.Graph, inf_df: pd.DataFrame, out_path: Path):
    pos = _positions(G)
    node2bt = dict(zip(inf_df["node"], inf_df["betweenness"]))
    node2pr = dict(zip(inf_df["node"], inf_df["pagerank"]))
    sizes = _scale([node2bt.get(n, 0.0) for n in G.nodes()], min_size=20, max_size=240)
    pr_vals = pd.Series([node2pr.get(n, 0.0) for n in G.nodes()], dtype=float)
    norm = colors.Normalize(vmin=pr_vals.min(), vmax=pr_vals.max() if pr_vals.max()>pr_vals.min() else pr_vals.min()+1)
    cmap = cm.viridis
    node_colors = [cmap(norm(node2pr.get(n, 0.0))) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_edges_sampled(G, pos)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes, linewidths=0.2, edgecolors="#ffffff")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="PageRank (color)")
    ax.set_title("Influence View (size=betweenness, color=pagerank)")
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_link_analysis_view(G: nx.Graph, link_df: pd.DataFrame, out_path: Path):
    pos = _positions(G)
    node2pr = dict(zip(link_df["node"], link_df["pagerank"]))
    node2eig = dict(zip(link_df["node"], link_df["eigenvector"]))
    sizes = _scale([node2pr.get(n, 0.0) for n in G.nodes()], min_size=20, max_size=240)
    eig_vals = pd.Series([node2eig.get(n, 0.0) for n in G.nodes()], dtype=float)
    norm = colors.Normalize(vmin=eig_vals.min(), vmax=eig_vals.max() if eig_vals.max()>eig_vals.min() else eig_vals.min()+1)
    cmap = cm.plasma
    node_colors = [cmap(norm(node2eig.get(n, 0.0))) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_edges_sampled(G, pos)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes, linewidths=0.2, edgecolors="#ffffff")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Eigenvector centrality (color)")
    ax.set_title("Link Analysis View (size=pagerank, color=eigenvector)")
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_anomaly_view(G: nx.Graph, anom_df: pd.DataFrame, out_path: Path):
    pos = _positions(G)
    node2anom = dict(zip(anom_df["node"], anom_df["decision_function"]))
    # invert so higher color = more anomalous
    vals = pd.Series([-node2anom.get(n, 0.0) for n in G.nodes()], dtype=float)
    norm = colors.Normalize(vmin=vals.min(), vmax=vals.max() if vals.max()>vals.min() else vals.min()+1)
    cmap = cm.coolwarm
    node_colors = [cmap(norm(-node2anom.get(n, 0.0))) for n in G.nodes()]
    sizes = _scale([G.degree(n) for n in G.nodes()], min_size=10, max_size=180)

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_edges_sampled(G, pos)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes, linewidths=0.2, edgecolors="#ffffff")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Anomaly (higher=more anomalous)")
    ax.set_title("Anomaly View (color=anomaly score, size=degree)")
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_views():
    G = nx.read_edgelist(DATA_PATH, create_using=nx.Graph(), nodetype=int)
    labels = pd.read_csv(OUTDIR / "node_classification_labels.csv")
    link = pd.read_csv(OUTDIR / "link_analysis.csv")
    influence = pd.read_csv(OUTDIR / "influence_scores.csv")
    anomalies = pd.read_csv(OUTDIR / "anomalies.csv")

    generate_community_view(G, labels, link, COMMUNITY_IMG)
    generate_link_analysis_view(G, link, LINK_ANALYSIS_IMG)
    generate_influence_view(G, influence, INFLUENCE_IMG)
    generate_anomaly_view(G, anomalies, ANOMALY_IMG)


def generate_roc_final():
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    scores_path = OUTDIR / "link_prediction_scores.csv"
    if not scores_path.exists():
        return
    df = pd.read_csv(scores_path)
    required = {"score", "metric", "label"}
    if not required.issubset(df.columns):
        return
    plt.figure(figsize=(7, 6))
    for metric, group in df.groupby("metric"):
        y_true = group["label"].astype(int).to_numpy()
        y_score = group["score"].astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{metric} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Link Prediction ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    outp = OUTDIR / "link_prediction_roc_final.png"
    plt.savefig(outp, dpi=150)
    plt.close()


# -------- PDF BUILDING ---------

def styleset():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleLarge', parent=styles['Title'], fontSize=26, leading=32, alignment=1, spaceAfter=18)
    subtitle = ParagraphStyle(name='Subtitle', parent=styles['Heading2'], fontSize=14, leading=18, alignment=1, textColor=rl_colors.grey)
    h2 = ParagraphStyle(name='H2', parent=styles['Heading2'], spaceBefore=12, spaceAfter=8)
    body = ParagraphStyle(name='Body', parent=styles['BodyText'], leading=14)
    small = ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=9, leading=12, textColor=rl_colors.grey)
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


def basic_stats_table() -> Table:
    G = nx.read_edgelist(DATA_PATH, create_using=nx.Graph(), nodetype=int)
    nodes = G.number_of_nodes(); edges = G.number_of_edges()
    density = nx.density(G); clustering = nx.average_clustering(G); is_conn = nx.is_connected(G)
    data = [["Metric", "Value"],["Nodes", f"{nodes}"],["Edges", f"{edges}"],["Density", f"{density:.4f}"],["Avg clustering", f"{clustering:.4f}"],["Connected", f"{is_conn}"]]
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.black),('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
        ('GRID', (0,0), (-1,-1), 0.3, rl_colors.grey),
    ]))
    return t


def build_pdf():
    ensure_outputs_exist()
    generate_views()
    generate_roc_final()

    title_style, subtitle, h2, body, small = styleset()

    def _footer(canvas, doc):
        canvas.saveState()
        page = canvas.getPageNumber()
        footer_text = f"{SUBJECT} — {STUDENT_NAME} (Roll: {ROLL_NO}) — Page {page}"
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(rl_colors.grey)
        canvas.drawRightString(A4[0] - 36, 20, footer_text)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    flow: List = []

    # Cover
    flow.append(Spacer(1, 1.0 * inch))
    flow.append(Paragraph(SUBJECT, title_style))
    flow.append(Paragraph("Final Project Report", subtitle))
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
    ], colWidths=[1.8*inch, 4.5*inch])
    meta_tbl.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, rl_colors.lightgrey),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [rl_colors.white, rl_colors.HexColor('#fbfbfb')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    flow.append(meta_tbl)
    flow.append(PageBreak())

    # 1. Overview
    flow.append(Paragraph("1. Dataset Overview", h2))
    flow.append(Paragraph("Email-EU-core undirected graph. Structural summary and degree characteristics.", body))
    flow.append(Spacer(1, 0.08 * inch))
    flow.append(Paragraph("Observations: The network is sparse (low density) and, as in many communication graphs, the degree distribution typically exhibits a heavy tail. A small set of nodes act as hubs with substantially higher degree, while most nodes have moderate to low degree. Average clustering indicates the tendency of colleagues to form tightly knit triads. If the graph is not fully connected, insights should be interpreted within the giant component.", body))
    flow.append(Paragraph("Interpretation: The subgraph snapshot provides intuition about the hub–periphery structure. The degree histogram helps motivate methods used later: centrality to identify important spreaders, community detection for group structure, and heuristics tailored to local neighborhoods for link prediction.", body))
    flow.append(Spacer(1, 0.1 * inch))
    flow.append(basic_stats_table())
    flow.append(Spacer(1, 0.15 * inch))
    add_image(flow, OUTDIR / "subgraph.png", "Figure 1: Subgraph visualization")
    add_image(flow, OUTDIR / "degree_distribution.png", "Figure 2: Degree distribution histogram")
    flow.append(PageBreak())

    # 2. Link Analysis
    flow.append(Paragraph("2. Link Analysis (PageRank & Eigenvector)", h2))
    flow.append(Paragraph("PageRank (importance via random walks) and Eigenvector centrality (importance via influential neighbors).", body))
    flow.append(Spacer(1, 0.08 * inch))
    flow.append(Paragraph("Observations: PageRank elevates nodes that attract many paths, directly or via iterative reinforcement; Eigenvector centrality favors nodes connected to other well-connected nodes. Overlap between the two measures typically signals strong hubs embedded in a highly connected core. Disagreements often reveal local elites or structurally peripheral nodes endorsed by a single dominant neighbor.", body))
    flow.append(Paragraph("Interpretation: In the network view, node size tracks PageRank while color follows an eigenvector gradient. Warm colors indicate high eigenvector centrality; large, warm nodes are the core influencers. The top PageRank bar chart complements the map by listing specific IDs you’d prioritize for information diffusion or monitoring.", body))
    la = pd.read_csv(OUTDIR / "link_analysis.csv").sort_values("pagerank", ascending=False)
    data = [list(["node","pagerank","eigenvector","degree"])] + la[["node","pagerank","eigenvector","degree"]].head(10).values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.black),('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey)
    ]))
    flow.append(table)
    add_image(flow, LINK_ANALYSIS_IMG, "Figure 3: Link Analysis view (size=pagerank, color=eigenvector)")
    add_image(flow, OUTDIR / "link_analysis_top_pagerank.png", "Figure 4: Top nodes by PageRank (bar chart)")
    flow.append(PageBreak())

    # 3. Node Classification
    flow.append(Paragraph("3. Node Classification (Label Propagation)", h2))
    flow.append(Paragraph("Asynchronous label propagation communities.", body))
    flow.append(Spacer(1, 0.08 * inch))
    flow.append(Paragraph("Observations: Label propagation uncovers cohesive groups that likely correspond to organizational units, projects, or frequent correspondents. A few large communities typically account for most nodes, with several smaller groups at the periphery.", body))
    flow.append(Paragraph("Interpretation: The distribution of community sizes suggests modular structure. Visual clusters in the community map align with these sizes; boundaries are porous where bridging nodes connect two modules—these nodes often reappear with high betweenness in the influence analysis.", body))
    labels = pd.read_csv(OUTDIR / "node_classification_labels.csv")
    comm_counts = labels.groupby("community").size().reset_index(name="size").sort_values("size", ascending=False)
    data = [list(["community","size"])] + comm_counts.head(15).values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.black),('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey)
    ]))
    flow.append(table)
    add_image(flow, COMMUNITY_IMG, "Figure 5: Communities (Python-generated equivalent)")
    flow.append(PageBreak())

    # 4. Influence Analysis
    flow.append(Paragraph("4. Influence Analysis (PageRank & Betweenness)", h2))
    flow.append(Paragraph("PageRank for global influence; Betweenness for brokerage across communities.", body))
    flow.append(Spacer(1, 0.08 * inch))
    flow.append(Paragraph("Observations: High PageRank nodes generally sit in the dense core, while top betweenness nodes are often boundary spanners connecting modules. When a node ranks highly on both, it’s a critical actor for both diffusion and bridging.", body))
    flow.append(Paragraph("Interpretation: Use the top-influencers chart to identify specific candidates for targeted messaging. In the influence map, larger nodes (betweenness) with strong color (PageRank) warrant special attention as both gatekeepers and amplifiers.", body))
    inf = pd.read_csv(OUTDIR / "influence_scores.csv").sort_values(["pagerank","betweenness"], ascending=False)
    data = [list(["node","pagerank","betweenness","degree"])] + inf.head(10).values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.black),('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey)
    ]))
    flow.append(table)
    add_image(flow, OUTDIR / "influence_top.png", "Figure 6: Top influencers (bar/line)")
    add_image(flow, INFLUENCE_IMG, "Figure 7: Influence view (size=betweenness, color=pagerank)")
    flow.append(PageBreak())

    # 5. Link Prediction
    flow.append(Paragraph("5. Link Prediction (Adamic-Adar, Jaccard, Preferential Attachment)", h2))
    flow.append(Paragraph("Hold-out 10% edges; compute heuristic scores on train graph; evaluate ROC/AUC.", body))
    flow.append(Spacer(1, 0.08 * inch))
    flow.append(Paragraph("Observations: Local-neighborhood metrics like Adamic–Adar often perform strongly in social graphs where shared neighbors are informative. Jaccard rewards exclusive overlap, while Preferential Attachment favors globally high-degree pairs—useful when growth is driven by popularity.", body))
    flow.append(Paragraph("Interpretation: The ROC curves and AUC table summarize ranking quality across thresholds. The best curve bows furthest toward the top-left. Differences between metrics suggest whether closure (common neighbors) or popularity (degree) drives new links in this network.", body))
    auc = pd.read_csv(OUTDIR / "link_prediction_auc.csv").sort_values("auc", ascending=False)
    data = [list(["metric","auc"])] + auc.values.tolist()
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.black),('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey)
    ]))
    flow.append(table)
    roc_path = OUTDIR / "link_prediction_roc_final.png"
    if not roc_path.exists():
        roc_path = OUTDIR / "link_prediction_roc.png"
    add_image(flow, roc_path, "Figure 8: ROC curves for link prediction")
    flow.append(PageBreak())

    # 6. Anomaly Detection
    flow.append(Paragraph("6. Anomaly Detection (IsolationForest on egonet features)", h2))
    flow.append(Paragraph("Features: degree, clustering, average neighbor degree, egonet edges; Model: IsolationForest.", body))
    flow.append(Spacer(1, 0.08 * inch))
    flow.append(Paragraph("Observations: The model flags structurally unusual nodes—e.g., high degree but low clustering (broadcast hubs), or low degree with unexpectedly high clustering (insular ties). These outliers can reflect unique roles, errors, or atypical communication patterns.", body))
    flow.append(Paragraph("Interpretation: Treat anomalies as hypotheses for follow-up, not conclusions. Cross-reference with domain knowledge (department, role, time) to determine whether they are benign (e.g., mailing lists) or risk-relevant (e.g., chokepoints or isolated actors).", body))
    anom = pd.read_csv(OUTDIR / "anomalies.csv")
    top = anom.sort_values("decision_function").head(10)
    data = [list(["node","degree","clustering","avg_neighbor_degree","ego_edges","decision_function"])] + top.values.tolist()
    table = Table(data, hAlign='CENTER')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.black),('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey)
    ]))
    flow.append(table)
    add_image(flow, OUTDIR / "anomaly_scatter.png", "Figure 9: Anomaly scatter (Degree vs Clustering)")
    add_image(flow, ANOMALY_IMG, "Figure 10: Anomaly view (color=anomaly score, size=degree)")
    # End of report: no trailing page break

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    print(f"✅ PDF created: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
