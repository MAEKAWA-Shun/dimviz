
# t-SNE Visualizer (Streamlit + uv + Docker)

## Features
- Upload **CSV/XLSX**
- Inspect data, **drop selected columns & rows**
- Choose an optional **label column** (excluded from features; used for color)
- Run **t-SNE** with sliders for hyperparameters
- Plotly scatter (interactive hover, **zoom/pan disabled**)
- Download results (CSV, HTML)

## Run with uv
```bash
uv sync
uv run streamlit run app.py
```

## Run with pip (alt.)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Docker
```bash
docker build -t tsne-visualizer .
docker run -p 8501:8501 tsne-visualizer
```
