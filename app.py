"""
CVPR Explorer — interactive 2D paper browser.

Data pipeline (run once before starting the app):
    python download.py --conf CVPR --year 2024
    python embed.py --conf CVPR --year 2024

Then start the app:
    python app.py
    # or via gunicorn:
    gunicorn app:server -b 0.0.0.0:8050
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
from dotenv import load_dotenv
from openai import OpenAI
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
N_CLUSTERS = 20          # number of semantic clusters per dataset
LLM_MODEL = "claude-opus-4-8"
LLM_FALLBACK = "claude-sonnet-4-6"

# Palette — distinct colours for up to N_CLUSTERS clusters
CLUSTER_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#9b5de5", "#06d6a0", "#ef476f", "#118ab2", "#ffd166",
    "#06a77d", "#d62828", "#023e8a", "#80b918", "#e76f51",
    "#a8dadc", "#c77dff", "#ff9e00", "#4cc9f0", "#f72585",
]


# ── LLM cluster labelling ─────────────────────────────────────────────────────

def make_llm_client() -> OpenAI | None:
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY", "placeholder")
    if not base_url:
        return None
    return OpenAI(base_url=base_url, api_key=api_key)


def label_cluster(client: OpenAI, titles: list[str], model: str) -> str:
    titles_block = "\n".join(f"- {t}" for t in titles)
    prompt = (
        "Below are titles of computer vision research papers that form a semantic cluster.\n"
        "Reply with ONLY a short label (3-6 words) capturing the common research theme.\n"
        "No explanation, no punctuation at the end.\n\n"
        f"{titles_block}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def compute_cluster_labels(df: pd.DataFrame, client: OpenAI | None) -> dict[int, str]:
    labels: dict[int, str] = {}
    if client is None:
        for cid in sorted(df["cluster"].unique()):
            labels[cid] = f"Cluster {cid}"
        return labels

    cluster_ids = sorted(df["cluster"].unique())
    print(f"  Labelling {len(cluster_ids)} clusters via LLM...")
    for cid in cluster_ids:
        titles = df[df["cluster"] == cid]["title"].tolist()
        for model in (LLM_MODEL, LLM_FALLBACK):
            try:
                label = label_cluster(client, titles, model)
                labels[cid] = label
                print(f"    Cluster {cid:2d} ({len(titles):4d} papers): {label}")
                break
            except Exception as e:
                print(f"    Cluster {cid} failed with {model}: {e}")
                time.sleep(1)
        else:
            labels[cid] = f"Cluster {cid}"

    return labels


# ── Convex hull cloud ─────────────────────────────────────────────────────────

def hull_trace(
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
    label: str,
    cluster_id: int,
) -> go.Scatter | None:
    points = np.column_stack([xs, ys])
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        verts = hull.vertices
        hx = np.append(points[verts, 0], points[verts[0], 0])
        hy = np.append(points[verts, 1], points[verts[0], 1])
    except Exception:
        return None

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return go.Scatter(
        x=hx,
        y=hy,
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.10)",
        line=dict(color=f"rgba({r},{g},{b},0.55)", width=1.5),
        mode="lines",
        name=label,
        legendgroup=f"cluster_{cluster_id}",
        showlegend=False,
        hoverinfo="skip",
    )


# ── Data loading ─────────────────────────────────────────────────────────────

def available_datasets() -> list[tuple[str, int]]:
    found = []
    for p in sorted(DATA_DIR.glob("*_papers.json")):
        parts = p.stem.split("_")
        if len(parts) >= 2:
            conf = parts[0].upper()
            try:
                found.append((conf, int(parts[1])))
            except ValueError:
                pass
    return found


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    if HAS_UMAP and len(embeddings) >= 10:
        reducer = umap.UMAP(
            n_components=2, random_state=42,
            metric="cosine", n_neighbors=15, min_dist=0.1,
        )
        return reducer.fit_transform(embeddings)
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(
        n_components=2, random_state=42,
        perplexity=perplexity, n_iter=1000, metric="cosine",
    )
    return tsne.fit_transform(embeddings)


def load_dataset(conf: str, year: int, llm_client: OpenAI | None) -> pd.DataFrame | None:
    papers_path = DATA_DIR / f"{conf.lower()}_{year}_papers.json"
    emb_path    = DATA_DIR / f"{conf.lower()}_{year}_embeddings.npy"
    labels_path = DATA_DIR / f"{conf.lower()}_{year}_cluster_labels.json"

    if not papers_path.exists():
        return None

    with open(papers_path) as f:
        papers = json.load(f)

    df = pd.DataFrame(papers)
    df["conf"] = conf
    df["year"] = year

    if emb_path.exists():
        embeddings = np.load(emb_path)
        if len(embeddings) == len(df):
            print(f"  Reducing {conf} {year} to 2D...")
            coords = reduce_to_2d(embeddings)
            df["x"] = coords[:, 0]
            df["y"] = coords[:, 1]
        else:
            df["x"] = np.random.randn(len(df))
            df["y"] = np.random.randn(len(df))
    else:
        df["x"] = np.random.randn(len(df))
        df["y"] = np.random.randn(len(df))

    # ── K-means clustering on 2D coords ──
    n_clusters = min(N_CLUSTERS, len(df))
    coords_2d = StandardScaler().fit_transform(df[["x", "y"]].values)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(coords_2d)

    # ── LLM cluster labels (cached to disk) ──
    if labels_path.exists():
        with open(labels_path) as f:
            raw = json.load(f)
        cluster_labels = {int(k): v for k, v in raw.items()}
    else:
        cluster_labels = compute_cluster_labels(df, llm_client)
        with open(labels_path, "w") as f:
            json.dump({str(k): v for k, v in cluster_labels.items()}, f, indent=2)

    df["cluster_label"] = df["cluster"].map(cluster_labels)
    return df


# ── Startup ───────────────────────────────────────────────────────────────────

print("Loading datasets...")
llm_client = make_llm_client()
if llm_client is None:
    print("  Warning: API_BASE_URL not set — cluster labels will be generic")

DATASETS: dict[tuple[str, int], pd.DataFrame] = {}
CLUSTER_LABELS: dict[tuple[str, int], dict[int, str]] = {}

for conf, year in available_datasets():
    print(f"  {conf} {year}...", flush=True)
    df = load_dataset(conf, year, llm_client)
    if df is not None:
        DATASETS[(conf, year)] = df
        CLUSTER_LABELS[(conf, year)] = dict(
            zip(df["cluster"], df["cluster_label"])
        )
        print(f"    {len(df)} papers, {df['cluster'].nunique()} clusters")

if not DATASETS:
    print("No datasets found — launching with demo data.")
    demo_df = pd.DataFrame({
        "title":         ["Demo Paper 1", "Demo Paper 2"],
        "abstract":      ["Demo abstract.", "Another demo."],
        "pdf_link":      ["", ""],
        "conf":          ["DEMO", "DEMO"],
        "year":          [2024, 2024],
        "cluster":       [0, 1],
        "cluster_label": ["Demo Cluster A", "Demo Cluster B"],
        "x":             [0.0, 1.0],
        "y":             [0.0, 1.0],
    })
    DATASETS[("DEMO", 2024)] = demo_df
    CLUSTER_LABELS[("DEMO", 2024)] = {0: "Demo Cluster A", 1: "Demo Cluster B"}

dataset_options = [
    {"label": f"{conf} {year}", "value": f"{conf}_{year}"}
    for conf, year in sorted(DATASETS.keys())
]
default_dataset = dataset_options[0]["value"]


# ── Dash layout ───────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="CVPR Explorer",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif", "background": "#0f1117", "minHeight": "100vh"},
    children=[
        # Header
        html.Div(
            style={
                "background": "#1a1d27", "padding": "16px 24px",
                "display": "flex", "alignItems": "center",
                "justifyContent": "space-between", "borderBottom": "1px solid #2d3148",
            },
            children=[
                html.H1("CVPR Explorer",
                        style={"color": "#e2e8f0", "margin": 0, "fontSize": "22px", "fontWeight": "700"}),
                html.Span("Browse computer vision papers by semantic similarity",
                          style={"color": "#8892b0", "fontSize": "13px"}),
            ],
        ),

        # Controls
        html.Div(
            style={
                "background": "#13161f", "padding": "12px 24px",
                "display": "flex", "gap": "16px", "flexWrap": "wrap",
                "alignItems": "center", "borderBottom": "1px solid #2d3148",
            },
            children=[
                html.Div([
                    html.Label("Conference / Year",
                               style={"color": "#8892b0", "fontSize": "11px", "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="dataset-dropdown", options=dataset_options,
                        value=default_dataset, clearable=False,
                        style={"width": "160px", "fontSize": "13px"},
                    ),
                ]),
                html.Div([
                    html.Label("Keyword Search",
                               style={"color": "#8892b0", "fontSize": "11px", "marginBottom": "4px", "display": "block"}),
                    dcc.Input(
                        id="search-input", type="text",
                        placeholder="e.g. diffusion model...", debounce=True,
                        style={
                            "width": "240px", "padding": "7px 12px",
                            "background": "#1a1d27", "border": "1px solid #2d3148",
                            "borderRadius": "6px", "color": "#e2e8f0", "fontSize": "13px",
                        },
                    ),
                ]),
                html.Div(
                    id="paper-count",
                    style={"color": "#8892b0", "fontSize": "12px", "marginLeft": "auto", "alignSelf": "flex-end"},
                ),
            ],
        ),

        # Main
        html.Div(
            style={"display": "flex", "height": "calc(100vh - 115px)"},
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": 0},
                    children=[dcc.Graph(
                        id="scatter-plot", style={"height": "100%"},
                        config={"displayModeBar": True, "scrollZoom": True},
                    )],
                ),
                html.Div(
                    id="detail-panel",
                    style={
                        "width": "360px", "background": "#1a1d27",
                        "borderLeft": "1px solid #2d3148", "padding": "20px",
                        "overflowY": "auto", "display": "flex",
                        "flexDirection": "column", "gap": "12px",
                    },
                    children=[html.P(
                        "Click a point on the map to see paper details.",
                        style={"color": "#4a5568", "fontSize": "13px",
                               "textAlign": "center", "marginTop": "40px"},
                    )],
                ),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("scatter-plot", "figure"),
    Output("paper-count", "children"),
    Input("dataset-dropdown", "value"),
    Input("search-input", "value"),
)
def update_scatter(dataset_value: str, search_query: str):
    conf, year_str = dataset_value.split("_", 1)
    year = int(year_str)
    df_full = DATASETS.get((conf, year), pd.DataFrame())

    # Keyword filter (highlights matching papers; others shown dimmed)
    if search_query and search_query.strip():
        pattern = re.escape(search_query.strip())
        mask = (
            df_full["title"].str.contains(pattern, case=False, na=False) |
            df_full["abstract"].str.contains(pattern, case=False, na=False)
        )
        df = df_full[mask]
    else:
        df = df_full

    traces: list = []
    cluster_ids = sorted(df_full["cluster"].unique())

    for i, cid in enumerate(cluster_ids):
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        label = CLUSTER_LABELS.get((conf, year), {}).get(cid, f"Cluster {cid}")
        sub_full = df_full[df_full["cluster"] == cid]
        sub = df[df["cluster"] == cid]

        # Convex hull cloud over the full cluster (always drawn)
        cloud = hull_trace(sub_full["x"].values, sub_full["y"].values, color, label, cid)
        if cloud is not None:
            traces.append(cloud)

        # Cluster centroid label annotation added via layout.annotations below
        if len(sub) == 0:
            continue

        hover_text = [
            f"<b>{row['title']}</b><br><i>{label}</i>"
            for _, row in sub.iterrows()
        ]
        traces.append(go.Scattergl(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=label,
            legendgroup=f"cluster_{cid}",
            marker=dict(size=5, color=color, opacity=0.8, line=dict(width=0)),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=sub.index.tolist(),
        ))

    # Centroid annotations
    annotations = []
    for i, cid in enumerate(cluster_ids):
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        label = CLUSTER_LABELS.get((conf, year), {}).get(cid, f"Cluster {cid}")
        sub_full = df_full[df_full["cluster"] == cid]
        cx, cy = sub_full["x"].mean(), sub_full["y"].mean()
        annotations.append(dict(
            x=cx, y=cy,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color=color, family="Inter, system-ui, sans-serif"),
            bgcolor="rgba(15,17,23,0.65)",
            borderpad=3,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e2e8f0", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            bgcolor="rgba(26,29,39,0.9)", bordercolor="#2d3148",
            borderwidth=1, font=dict(size=10),
            itemclick="toggleothers", itemdoubleclick="toggle",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        dragmode="pan",
        uirevision=dataset_value,
        annotations=annotations,
    )

    count_str = f"{len(df):,} of {len(df_full):,} papers shown"
    return fig, count_str


@app.callback(
    Output("detail-panel", "children"),
    Input("scatter-plot", "clickData"),
    State("dataset-dropdown", "value"),
)
def show_paper_detail(click_data, dataset_value: str):
    if not click_data:
        return [html.P(
            "Click a point on the map to see paper details.",
            style={"color": "#4a5568", "fontSize": "13px",
                   "textAlign": "center", "marginTop": "40px"},
        )]

    point = click_data["points"][0]
    idx = point.get("customdata")

    conf, year_str = dataset_value.split("_", 1)
    year = int(year_str)
    df = DATASETS.get((conf, year), pd.DataFrame())

    if idx is None or idx not in df.index:
        return [html.P("Paper not found.", style={"color": "#e53e3e"})]

    row = df.loc[idx]
    cid = int(row["cluster"])
    label = CLUSTER_LABELS.get((conf, year), {}).get(cid, f"Cluster {cid}")
    color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
    pdf_link = row.get("pdf_link", "")
    abstract = row.get("abstract", "No abstract available.")

    children = [
        html.Div(label, style={
            "display": "inline-block",
            "background": color + "33",
            "color": color,
            "borderRadius": "4px",
            "padding": "2px 8px",
            "fontSize": "11px",
            "fontWeight": "600",
            "marginBottom": "8px",
        }),
        html.H3(row["title"], style={
            "color": "#e2e8f0", "fontSize": "15px", "fontWeight": "600",
            "lineHeight": "1.4", "margin": "0 0 12px 0",
        }),
        html.P(abstract, style={
            "color": "#a0aec0", "fontSize": "12px",
            "lineHeight": "1.6", "margin": "0 0 16px 0",
        }),
    ]

    if pdf_link:
        children.append(html.A(
            "Open PDF →", href=pdf_link, target="_blank",
            style={
                "display": "inline-block", "background": "#3182ce",
                "color": "#fff", "borderRadius": "6px",
                "padding": "8px 16px", "fontSize": "13px",
                "fontWeight": "600", "textDecoration": "none",
            },
        ))

    return children


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
