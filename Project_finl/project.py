from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import networkx as nx


def load_graph() -> nx.Graph:
	script_dir = Path(__file__).resolve().parent
	data_path = script_dir / "email-Eu-core.txt"
	G = nx.read_edgelist(data_path, create_using=nx.Graph(), nodetype=int)
	return G


def ensure_backend(show: bool):
	if show:
		import matplotlib.pyplot as plt  # noqa: WPS433
	else:
		import matplotlib  # noqa: WPS433
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt  # noqa: WPS433
	return plt


def task_overview(G: nx.Graph, show: bool, outdir: Path, subgraph_png: str, degree_png: str, sample_n: int) -> None:
	plt = ensure_backend(show)

	print("✅ Graph loaded successfully!")
	print(f"Nodes: {G.number_of_nodes()}")
	print(f"Edges: {G.number_of_edges()}")

	# Figure 1: subgraph
	plt.figure(figsize=(8, 6))
	nodes = list(G.nodes())[: max(0, sample_n)]
	nx.draw(G.subgraph(nodes), with_labels=False, node_size=50)
	plt.title("Email-EU-core: Subgraph Visualization")

	# Basic stats
	avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
	print(f"Average Degree: {avg_degree:.2f}")
	density = nx.density(G)
	print(f"Density: {density:.4f}")
	avg_clustering = nx.average_clustering(G)
	print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
	is_conn = nx.is_connected(G)
	print(f"Is Graph Connected?: {is_conn}")
	if is_conn:
		diameter = nx.diameter(G)
		print(f"Network Diameter: {diameter}")
	else:
		print("Graph is not fully connected, so diameter not available.")

	# Figure 2: degree distribution
	degrees = [d for _, d in G.degree()]
	plt.figure(figsize=(8, 6))
	plt.hist(degrees, bins=50, color="#4C72B0", edgecolor="white")
	plt.xlabel("Degree")
	plt.ylabel("Frequency")
	plt.title("Email-EU-core: Degree Distribution")

	if show:
		plt.show()
	else:
		outdir.mkdir(parents=True, exist_ok=True)
		plt.figure(1)
		p1 = outdir / subgraph_png
		plt.savefig(p1, bbox_inches="tight", dpi=150)
		print(f"📸 Saved visualization to: {p1}")

		plt.figure(2)
		p2 = outdir / degree_png
		plt.savefig(p2, bbox_inches="tight", dpi=150)
		print(f"📊 Saved degree distribution to: {p2}")


def task_link_analysis(G: nx.Graph, show: bool, outdir: Path, topk: int = 20) -> None:
	import pandas as pd
	plt = ensure_backend(show)

	print("▶️ Link analysis: PageRank and Eigenvector centrality")
	pr = nx.pagerank(G)
	ev = nx.eigenvector_centrality(G, max_iter=1000)

	df = pd.DataFrame({
		"node": list(pr.keys()),
		"pagerank": list(pr.values()),
		"eigenvector": [ev[n] for n in pr.keys()],
		"degree": [G.degree(n) for n in pr.keys()],
	}).sort_values("pagerank", ascending=False)

	outdir.mkdir(parents=True, exist_ok=True)
	df.to_csv(outdir / "link_analysis.csv", index=False)

	# Bar chart of top-k by PageRank
	top = df.head(topk)
	plt.figure(figsize=(10, 5))
	plt.bar(top["node"].astype(str), top["pagerank"], color="#55A868")
	plt.xticks(rotation=90)
	plt.title("Top nodes by PageRank")
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.savefig(outdir / "link_analysis_top_pagerank.png", dpi=150)
		print(f"📈 Saved: {outdir / 'link_analysis_top_pagerank.png'}")

	# Attach attributes for Gephi export
	nx.set_node_attributes(G, pr, name="pagerank")
	nx.set_node_attributes(G, ev, name="eigenvector")


def task_node_classification(G: nx.Graph, outdir: Path) -> None:
	print("▶️ Node classification via Label Propagation (communities)")
	# Community detection using asynchronous Label Propagation
	comms = list(nx.algorithms.community.asyn_lpa_communities(G, weight=None, seed=42))
	node2comm = {}
	for cid, nodes in enumerate(comms):
		for n in nodes:
			node2comm[n] = cid

	import pandas as pd
	df = pd.DataFrame({"node": list(node2comm.keys()), "community": list(node2comm.values())})
	outdir.mkdir(parents=True, exist_ok=True)
	df.to_csv(outdir / "node_classification_labels.csv", index=False)
	print(f"🗂️ Saved communities for {len(df)} nodes across {len(comms)} communities")

	nx.set_node_attributes(G, node2comm, name="community")


def task_influence(G: nx.Graph, show: bool, outdir: Path, topk: int = 20) -> None:
	import pandas as pd
	plt = ensure_backend(show)

	print("▶️ Influence analysis: PageRank and Betweenness centrality")
	pr = nx.pagerank(G)
	bt = nx.betweenness_centrality(G, normalized=True)

	df = pd.DataFrame({
		"node": list(pr.keys()),
		"pagerank": list(pr.values()),
		"betweenness": [bt[n] for n in pr.keys()],
		"degree": [G.degree(n) for n in pr.keys()],
	}).sort_values(["pagerank", "betweenness"], ascending=False)

	outdir.mkdir(parents=True, exist_ok=True)
	df.to_csv(outdir / "influence_scores.csv", index=False)

	top = df.head(topk)
	plt.figure(figsize=(10, 5))
	plt.bar(top["node"].astype(str), top["pagerank"], label="PageRank", color="#4C72B0")
	plt.plot(top["node"].astype(str), top["betweenness"], "o-", color="#C44E52", label="Betweenness")
	plt.xticks(rotation=90)
	plt.legend()
	plt.title("Top influencers")
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.savefig(outdir / "influence_top.png", dpi=150)
		print(f"🚀 Saved: {outdir / 'influence_top.png'}")

	nx.set_node_attributes(G, bt, name="betweenness")


def _nonedge_samples(G: nx.Graph, n: int, seed: int = 42) -> Iterable[Tuple[int, int]]:
	import random
	random.seed(seed)
	non_edges = list(nx.non_edges(G))
	if n > len(non_edges):
		n = len(non_edges)
	return random.sample(non_edges, n)


def task_link_prediction(G: nx.Graph, show: bool, outdir: Path, holdout_frac: float = 0.1, seed: int = 42) -> None:
	from sklearn.metrics import roc_auc_score, RocCurveDisplay
	import pandas as pd
	plt = ensure_backend(show)
	import random

	print("▶️ Link prediction: Adamic-Adar, Jaccard, Preferential Attachment")
	outdir.mkdir(parents=True, exist_ok=True)

	# Split edges into train and test (holdout)
	edges = list(G.edges())
	random.seed(seed)
	random.shuffle(edges)
	split = int((1 - holdout_frac) * len(edges))
	train_edges = edges[:split]
	test_edges = edges[split:]

	G_train = nx.Graph()
	G_train.add_nodes_from(G.nodes())
	G_train.add_edges_from(train_edges)

	# Sample negative edges equal to test size
	neg_samples = list(_nonedge_samples(G_train, len(test_edges), seed=seed))

	# Prepare ebunch for the union of positives and negatives
	pos_pairs = test_edges
	ebunch = pos_pairs + neg_samples

	def score_to_df(name: str, scores_iter: Iterable[Tuple[int, int, float]]) -> pd.DataFrame:
		u, v, s = [], [], []
		for a, b, val in scores_iter:
			u.append(a); v.append(b); s.append(val)
		return pd.DataFrame({"u": u, "v": v, "score": s, "metric": name})

	# Compute scores on G_train for all pairs in ebunch
	# Adamic-Adar can encounter log(1)=0 if a common neighbor has degree 1.
	# Use a safe variant that skips such neighbors.
	import math
	def safe_adamic_adar(graph: nx.Graph, pairs: Iterable[Tuple[int, int]]):
		for u, v in pairs:
			s = 0.0
			for w in nx.common_neighbors(graph, u, v):
				deg = graph.degree(w)
				if deg > 1:
					s += 1.0 / math.log(deg)
			yield (u, v, s)

	aa_scores = list(safe_adamic_adar(G_train, ebunch))
	jc_scores = list(nx.jaccard_coefficient(G_train, ebunch))
	pa_scores = list(nx.preferential_attachment(G_train, ebunch))

	import pandas as pd
	df_scores = pd.concat([
		score_to_df("adamic_adar", aa_scores),
		score_to_df("jaccard", jc_scores),
		score_to_df("pref_attach", pa_scores),
	], ignore_index=True)

	# Labels: 1 for held-out positives, 0 for negatives
	pos_set = {tuple(sorted(e)) for e in pos_pairs}
	df_scores["label"] = [1 if tuple(sorted((u, v))) in pos_set else 0 for u, v in zip(df_scores["u"], df_scores["v"])]

	# Compute AUC and plot ROC for each metric
	plt.figure(figsize=(7, 6))
	aucs = {}
	for metric, group in df_scores.groupby("metric"):
		y_true = group["label"].values
		y_score = group["score"].values
		auc = roc_auc_score(y_true, y_score)
		aucs[metric] = auc
		RocCurveDisplay.from_predictions(y_true, y_score, name=f"{metric} (AUC={auc:.3f})")
	plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
	plt.title("Link Prediction ROC Curves")
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.savefig(outdir / "link_prediction_roc.png", dpi=150)
		print(f"🧪 Saved: {outdir / 'link_prediction_roc.png'}")

	# Save scores and summary
	df_scores.to_csv(outdir / "link_prediction_scores.csv", index=False)
	pd.DataFrame([{"metric": k, "auc": v} for k, v in aucs.items()]).to_csv(outdir / "link_prediction_auc.csv", index=False)


def task_anomaly_detection(G: nx.Graph, show: bool, outdir: Path, contamination: float = 0.01) -> None:
	import numpy as np
	import pandas as pd
	from sklearn.ensemble import IsolationForest
	plt = ensure_backend(show)

	print("▶️ Anomaly detection: IsolationForest on egonet features")

	# Features per node: degree, clustering, avg neighbor degree, egonet edges
	degrees = dict(G.degree())
	clustering = nx.clustering(G)
	avg_neigh_deg = nx.average_neighbor_degree(G)

	def egonet_edges_count(n: int) -> int:
		ego = nx.ego_graph(G, n)
		return ego.number_of_edges()

	ego_edges = {n: egonet_edges_count(n) for n in G.nodes()}

	df = pd.DataFrame({
		"node": list(G.nodes()),
		"degree": [degrees[n] for n in G.nodes()],
		"clustering": [clustering[n] for n in G.nodes()],
		"avg_neighbor_degree": [avg_neigh_deg[n] for n in G.nodes()],
		"ego_edges": [ego_edges[n] for n in G.nodes()],
	})

	X = df[["degree", "clustering", "avg_neighbor_degree", "ego_edges"]].values.astype(float)
	iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
	df["anomaly_score"] = -iso.fit_predict(X)  # 1 (inlier) -> -1, so negate to make higher = more anomalous
	df["decision_function"] = iso.decision_function(X)

	outdir.mkdir(parents=True, exist_ok=True)
	df.sort_values(["anomaly_score", "decision_function"], ascending=False).to_csv(outdir / "anomalies.csv", index=False)

	# Scatter: degree vs clustering colored by anomaly score
	plt.figure(figsize=(8, 6))
	sc = plt.scatter(df["degree"], df["clustering"], c=df["decision_function"], cmap="coolwarm", s=20)
	plt.colorbar(sc, label="IsolationForest score (higher=more normal)")
	plt.xlabel("Degree")
	plt.ylabel("Clustering Coefficient")
	plt.title("Anomaly Detection: Degree vs Clustering")
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.savefig(outdir / "anomaly_scatter.png", dpi=150)
		print(f"🔎 Saved: {outdir / 'anomaly_scatter.png'}")

	# Attach attribute
	node2anom = {int(n): float(s) for n, s in zip(df["node"], df["decision_function"]) }
	nx.set_node_attributes(G, node2anom, name="anomaly_score")


def export_gexf(G: nx.Graph, outdir: Path, name: str = "graph_attributes.gexf") -> None:
	outdir.mkdir(parents=True, exist_ok=True)
	path = outdir / name
	nx.write_gexf(G, path)
	print(f"🗺️ Exported GEXF for Gephi: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="SNA Project Toolkit: run analyses and generate outputs.")
	p.add_argument("--task", choices=[
		"overview", "link_analysis", "node_classification", "influence", "link_prediction", "anomaly", "all"
	], default="all", help="Which analysis task to run")
	p.add_argument("--show", action="store_true", help="Show plots instead of saving")
	p.add_argument("--outdir", default="outputs", help="Directory to write outputs")
	p.add_argument("--out", default="subgraph.png", help="Subgraph image filename for overview task")
	p.add_argument("--out-degree", default="degree_distribution.png", help="Degree histogram image filename for overview task")
	p.add_argument("--sample-n", type=int, default=50, help="Nodes to draw in subgraph (overview)")
	return p


def main(argv: list[str] | None = None) -> int:
	args = build_arg_parser().parse_args(argv)
	outdir = Path(args.outdir)

	G = load_graph()

	# Run selected tasks
	if args.task in ("overview", "all"):
		task_overview(G, args.show, outdir, args.out, args.out_degree, args.sample_n)
	if args.task in ("link_analysis", "all"):
		task_link_analysis(G, args.show, outdir)
	if args.task in ("node_classification", "all"):
		task_node_classification(G, outdir)
	if args.task in ("influence", "all"):
		task_influence(G, args.show, outdir)
	if args.task in ("link_prediction", "all"):
		task_link_prediction(G, args.show, outdir)
	if args.task in ("anomaly", "all"):
		task_anomaly_detection(G, args.show, outdir)

	# Always export GEXF at the end so Gephi can use latest attributes
	export_gexf(G, outdir)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
