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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ── Field classification ─────────────────────────────────────────────────────

FIELD_KEYWORDS: dict[str, list[str]] = {
    "Diffusion / Generative": [
        "diffusion", "generative", "GAN", "flow matching", "score matching",
        "denoising", "stable diffusion", "text-to-image",
    ],
    "Vision Transformer": [
        "vision transformer", "ViT", "DEIT", "swin transformer", "attention",
        "self-attention", "token", "patch embedding",
    ],
    "NeRF / 3D": [
        "nerf", "neural radiance", "3d reconstruction", "point cloud",
        "mesh", "depth estimation", "multi-view", "gaussian splatting",
    ],
    "Object Detection": [
        "object detection", "YOLO", "DETR", "anchor", "bounding box",
        "proposal", "two-stage", "one-stage",
    ],
    "Segmentation": [
        "segmentation", "semantic segmentation", "instance segmentation",
        "panoptic", "SAM", "mask",
    ],
    "Video Understanding": [
        "video", "temporal", "action recognition", "optical flow",
        "tracking", "video understanding",
    ],
    "Multi-modal / VLM": [
        "CLIP", "multi-modal", "vision-language", "VLM", "LLaVA",
        "visual question", "image captioning", "grounding",
    ],
    "Medical Imaging": [
        "medical", "clinical", "pathology", "radiology", "MRI", "CT scan",
        "histology", "ultrasound",
    ],
    "Self-supervised / Contrastive": [
        "self-supervised", "contrastive", "representation learning",
        "pretraining", "MAE", "MoCo", "SimCLR",
    ],
    "Other": [],
}


def classify_paper(text: str) -> str:
    text_lower = text.lower()
    for field, keywords in FIELD_KEYWORDS.items():
        if field == "Other":
            continue
        if any(kw.lower() in text_lower for kw in keywords):
            return field
    return "Other"


# ── Data loading ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")

FIELD_COLORS = {
    "Diffusion / Generative": "#e63946",
    "Vision Transformer":     "#457b9d",
    "NeRF / 3D":              "#2a9d8f",
    "Object Detection":       "#e9c46a",
    "Segmentation":           "#f4a261",
    "Video Understanding":    "#9b5de5",
    "Multi-modal / VLM":      "#06d6a0",
    "Medical Imaging":        "#ef476f",
    "Self-supervised / Contrastive": "#118ab2",
    "Other":                  "#adb5bd",
}


def available_datasets() -> list[tuple[str, int]]:
    found = []
    for p in sorted(DATA_DIR.glob("*_papers.json")):
        stem = p.stem  # e.g. "cvpr_2024_papers"
        parts = stem.split("_")
        if len(parts) >= 2:
            conf = parts[0].upper()
            try:
                year = int(parts[1])
                found.append((conf, year))
            except ValueError:
                pass
    return found


def load_dataset(conf: str, year: int) -> pd.DataFrame | None:
    papers_path = DATA_DIR / f"{conf.lower()}_{year}_papers.json"
    emb_path = DATA_DIR / f"{conf.lower()}_{year}_embeddings.npy"

    if not papers_path.exists():
        return None

    with open(papers_path) as f:
        papers = json.load(f)

    df = pd.DataFrame(papers)
    df["conf"] = conf
    df["year"] = year
    df["field"] = df.apply(
        lambda r: classify_paper(str(r.get("abstract", "")) + " " + str(r.get("title", ""))),
        axis=1,
    )

    if emb_path.exists():
        embeddings = np.load(emb_path)
        if len(embeddings) == len(df):
            coords = reduce_to_2d(embeddings)
            df["x"] = coords[:, 0]
            df["y"] = coords[:, 1]
        else:
            df["x"] = np.random.randn(len(df))
            df["y"] = np.random.randn(len(df))
    else:
        # Fallback: random positions when no embeddings yet
        df["x"] = np.random.randn(len(df))
        df["y"] = np.random.randn(len(df))

    return df


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    if HAS_UMAP and len(embeddings) >= 10:
        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine",
                             n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    # Fall back to t-SNE
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                n_iter=1000, metric="cosine")
    return tsne.fit_transform(embeddings)


# Load all available datasets at startup
print("Loading datasets...")
DATASETS: dict[tuple[str, int], pd.DataFrame] = {}
for conf, year in available_datasets():
    print(f"  {conf} {year}...", end=" ", flush=True)
    df = load_dataset(conf, year)
    if df is not None:
        DATASETS[(conf, year)] = df
        print(f"{len(df)} papers")
    else:
        print("skipped")

if not DATASETS:
    print("No datasets found. Run download.py and embed.py first.")
    # Provide a tiny demo dataset so the app still launches
    demo_data = {
        "title": ["Demo Paper 1", "Demo Paper 2"],
        "abstract": ["This is a demo abstract.", "Another demo abstract."],
        "pdf_link": ["", ""],
        "conf": ["DEMO", "DEMO"],
        "year": [2024, 2024],
        "field": ["Other", "Other"],
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }
    DATASETS[("DEMO", 2024)] = pd.DataFrame(demo_data)

dataset_options = [
    {"label": f"{conf} {year}", "value": f"{conf}_{year}"}
    for conf, year in sorted(DATASETS.keys())
]
default_dataset = dataset_options[0]["value"] if dataset_options else "DEMO_2024"

# ── Dash app ─────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="CVPR Explorer",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose Flask server for gunicorn

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif", "background": "#0f1117", "minHeight": "100vh"},
    children=[
        # ── Header ──
        html.Div(
            style={
                "background": "#1a1d27",
                "padding": "16px 24px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "borderBottom": "1px solid #2d3148",
            },
            children=[
                html.H1(
                    "CVPR Explorer",
                    style={"color": "#e2e8f0", "margin": 0, "fontSize": "22px", "fontWeight": "700"},
                ),
                html.Span(
                    "Browse computer vision papers by semantic similarity",
                    style={"color": "#8892b0", "fontSize": "13px"},
                ),
            ],
        ),

        # ── Controls bar ──
        html.Div(
            style={
                "background": "#13161f",
                "padding": "12px 24px",
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "alignItems": "center",
                "borderBottom": "1px solid #2d3148",
            },
            children=[
                html.Div([
                    html.Label("Conference / Year", style={"color": "#8892b0", "fontSize": "11px", "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="dataset-dropdown",
                        options=dataset_options,
                        value=default_dataset,
                        clearable=False,
                        style={"width": "160px", "fontSize": "13px"},
                    ),
                ]),
                html.Div([
                    html.Label("Research Field", style={"color": "#8892b0", "fontSize": "11px", "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(
                        id="field-dropdown",
                        options=[{"label": "All Fields", "value": "all"}] +
                                [{"label": f, "value": f} for f in FIELD_KEYWORDS],
                        value="all",
                        clearable=False,
                        style={"width": "220px", "fontSize": "13px"},
                    ),
                ]),
                html.Div([
                    html.Label("Keyword Search", style={"color": "#8892b0", "fontSize": "11px", "marginBottom": "4px", "display": "block"}),
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="e.g. diffusion model...",
                        debounce=True,
                        style={
                            "width": "240px",
                            "padding": "7px 12px",
                            "background": "#1a1d27",
                            "border": "1px solid #2d3148",
                            "borderRadius": "6px",
                            "color": "#e2e8f0",
                            "fontSize": "13px",
                        },
                    ),
                ]),
                html.Div(
                    id="paper-count",
                    style={"color": "#8892b0", "fontSize": "12px", "marginLeft": "auto", "alignSelf": "flex-end"},
                ),
            ],
        ),

        # ── Main content ──
        html.Div(
            style={"display": "flex", "height": "calc(100vh - 130px)"},
            children=[
                # Scatter plot
                html.Div(
                    style={"flex": "1", "minWidth": 0},
                    children=[
                        dcc.Graph(
                            id="scatter-plot",
                            style={"height": "100%"},
                            config={"displayModeBar": True, "scrollZoom": True},
                        ),
                    ],
                ),

                # Paper detail panel
                html.Div(
                    id="detail-panel",
                    style={
                        "width": "360px",
                        "background": "#1a1d27",
                        "borderLeft": "1px solid #2d3148",
                        "padding": "20px",
                        "overflowY": "auto",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
                    children=[
                        html.P(
                            "Click a point on the map to see paper details.",
                            style={"color": "#4a5568", "fontSize": "13px", "textAlign": "center", "marginTop": "40px"},
                        )
                    ],
                ),
            ],
        ),
    ],
)


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("scatter-plot", "figure"),
    Output("paper-count", "children"),
    Input("dataset-dropdown", "value"),
    Input("field-dropdown", "value"),
    Input("search-input", "value"),
)
def update_scatter(dataset_value: str, field_filter: str, search_query: str):
    conf, year_str = dataset_value.split("_", 1)
    year = int(year_str)
    df = DATASETS.get((conf, year), pd.DataFrame())

    # Apply field filter
    if field_filter and field_filter != "all":
        df = df[df["field"] == field_filter]

    # Apply keyword search
    if search_query and search_query.strip():
        pattern = re.escape(search_query.strip())
        mask = (
            df["title"].str.contains(pattern, case=False, na=False) |
            df["abstract"].str.contains(pattern, case=False, na=False)
        )
        df = df[mask]

    traces = []
    fields_present = df["field"].unique() if len(df) else []

    for field in list(FIELD_KEYWORDS.keys()):
        if field not in fields_present:
            continue
        sub = df[df["field"] == field]
        hover_text = [
            f"<b>{row['title']}</b><br><i>{row['field']}</i>"
            for _, row in sub.iterrows()
        ]
        traces.append(
            go.Scattergl(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=field,
                marker=dict(
                    size=5,
                    color=FIELD_COLORS.get(field, "#adb5bd"),
                    opacity=0.75,
                    line=dict(width=0),
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                customdata=sub.index.tolist(),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e2e8f0", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            bgcolor="rgba(26,29,39,0.9)",
            bordercolor="#2d3148",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        dragmode="pan",
        uirevision=dataset_value,
    )

    count_str = f"{len(df):,} papers shown"
    return fig, count_str


@app.callback(
    Output("detail-panel", "children"),
    Input("scatter-plot", "clickData"),
    State("dataset-dropdown", "value"),
)
def show_paper_detail(click_data, dataset_value: str):
    if not click_data:
        return [
            html.P(
                "Click a point on the map to see paper details.",
                style={"color": "#4a5568", "fontSize": "13px", "textAlign": "center", "marginTop": "40px"},
            )
        ]

    point = click_data["points"][0]
    idx = point.get("customdata")

    conf, year_str = dataset_value.split("_", 1)
    year = int(year_str)
    df = DATASETS.get((conf, year), pd.DataFrame())

    if idx is None or idx not in df.index:
        return [html.P("Paper not found.", style={"color": "#e53e3e"})]

    row = df.loc[idx]
    field = row.get("field", "Other")
    pdf_link = row.get("pdf_link", "")
    abstract = row.get("abstract", "No abstract available.")

    children = [
        html.Div(
            field,
            style={
                "display": "inline-block",
                "background": FIELD_COLORS.get(field, "#adb5bd") + "33",
                "color": FIELD_COLORS.get(field, "#adb5bd"),
                "borderRadius": "4px",
                "padding": "2px 8px",
                "fontSize": "11px",
                "fontWeight": "600",
                "marginBottom": "8px",
            },
        ),
        html.H3(
            row["title"],
            style={"color": "#e2e8f0", "fontSize": "15px", "fontWeight": "600",
                   "lineHeight": "1.4", "margin": "0 0 12px 0"},
        ),
        html.P(
            abstract,
            style={"color": "#a0aec0", "fontSize": "12px", "lineHeight": "1.6",
                   "margin": "0 0 16px 0"},
        ),
    ]

    if pdf_link:
        children.append(
            html.A(
                "Open PDF →",
                href=pdf_link,
                target="_blank",
                style={
                    "display": "inline-block",
                    "background": "#3182ce",
                    "color": "#fff",
                    "borderRadius": "6px",
                    "padding": "8px 16px",
                    "fontSize": "13px",
                    "fontWeight": "600",
                    "textDecoration": "none",
                },
            )
        )

    return children


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
