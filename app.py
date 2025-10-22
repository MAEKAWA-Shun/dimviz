import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="t-SNE App", page_icon="🧭", layout="wide")


def get_df():
    return st.session_state.get("data_clean") or st.session_state.get("data_raw")


def save_df(df, key):
    st.session_state[key] = df


def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name is None:
        df = df.copy()
        df.index.name = "index"
    return df


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data upload", "t-SNE"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Session data status:")
if "data_clean" in st.session_state:
    st.sidebar.success("Cleaned data ready ✓")
elif "data_raw" in st.session_state:
    st.sidebar.info("Raw data loaded")
else:
    st.sidebar.warning("No data")

if page == "Data upload":
    st.title("Data upload")
    st.caption("Upload CSV or Excel, inspect rows/columns, and remove selected ones.")

    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    read_opts = {}

    if file is not None:
        if file.name.lower().endswith(".csv"):
            with st.expander("CSV read options"):
                sep = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
                encoding = st.text_input("Encoding (blank=auto)", value="")
                if sep:
                    read_opts["sep"] = sep
                if encoding.strip():
                    read_opts["encoding"] = encoding.strip()
            try:
                df = pd.read_csv(file, **read_opts)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df = None
        else:
            with st.expander("Excel read options"):
                sheet = st.text_input("Sheet name or index (blank = first)", value="")
            try:
                if sheet.strip() == "":
                    df = pd.read_excel(file, engine="openpyxl")
                else:
                    try:
                        idx = int(sheet)
                        df = pd.read_excel(file, sheet_name=idx, engine="openpyxl")
                    except Exception:
                        df = pd.read_excel(file, sheet_name=sheet, engine="openpyxl")
            except Exception as e:
                st.error(f"Failed to read Excel: {e}")
                df = None

        if df is not None:
            df = ensure_index(df)
            save_df(df, "data_raw")
            st.subheader("Preview")
            st.write("Shape:", df.shape)
            st.dataframe(df.head(50), use_container_width=True)

            st.markdown("### Column operations")
            col_drop = st.multiselect(
                "Select columns to drop", options=list(df.columns)
            )
            st.markdown("### Row operations")
            with st.expander("Select rows to drop"):
                idx_options = list(df.index)
                display_map = {str(i): i for i in idx_options}
                selected_display = st.multiselect(
                    "Select indices to drop", options=list(display_map.keys())
                )
                row_drop = [display_map[s] for s in selected_display]

            applied = st.button("Apply drops", type="primary")
            if applied:
                new_df = df.drop(columns=col_drop, errors="ignore").drop(
                    index=row_drop, errors="ignore"
                )
                save_df(new_df, "data_clean")
                st.success(
                    f"Applied: dropped {len(col_drop)} columns and {len(row_drop)} rows."
                )
                st.write("New shape:", new_df.shape)
                st.dataframe(new_df.head(50), use_container_width=True)

            if "data_clean" in st.session_state:
                out = st.session_state["data_clean"]
                st.download_button(
                    "Download cleaned CSV",
                    data=out.to_csv(index=True).encode("utf-8"),
                    file_name="cleaned.csv",
                    mime="text/csv",
                )

elif page == "t-SNE":
    st.title("t-SNE")
    df = get_df()
    if df is None:
        st.warning(
            "No data found. Please upload data on the **Data upload** page first."
        )
        st.stop()

    df = ensure_index(df)
    st.subheader("Data overview")
    with st.expander("Preview", expanded=True):
        st.write("Current shape:", df.shape)
        st.dataframe(df.head(30), use_container_width=True)

    st.markdown("### Label & features")
    all_columns = list(df.columns)
    label_col = st.selectbox(
        "Optional label column (excluded from features)",
        ["(none)"] + all_columns,
        index=0,
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_feats = [c for c in numeric_cols if c != label_col][
        : min(10, len(numeric_cols))
    ]
    feats = st.multiselect(
        "Numeric feature columns for t-SNE", options=numeric_cols, default=default_feats
    )

    st.markdown("### Preprocess")
    handle_na = st.selectbox(
        "Missing values", ["Drop rows with NA", "Impute with mean"], index=0
    )
    do_scale = st.checkbox("Standardize features", value=True)

    if len(feats) < 2:
        st.info("Select at least 2 numeric features to run t-SNE.")
        st.stop()

    X = df[feats].copy()
    if handle_na == "Drop rows with NA":
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        df_used = df.loc[X.index]
    else:
        X = X.fillna(X.mean(numeric_only=True))
        df_used = df

    if X.shape[0] < 5:
        st.error("Not enough rows after preprocessing (need ≥ 5).")
        st.stop()

    if do_scale:
        X = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=feats, index=X.index
        )

    st.subheader("Embedding (2D)")
    plot_area = st.empty()

    st.markdown("### t-SNE hyperparameters")
    n_samples = X.shape[0]
    max_perp = max(5, min(50, (n_samples - 1) // 3))
    perp = st.slider(
        "Perplexity",
        min_value=5,
        max_value=int(max_perp),
        value=int(min(30, max_perp)),
        step=1,
    )
    # n_iter = st.slider("Iterations (n_iter)", min_value=250, max_value=5000, value=1000, step=250)
    lr_choice = st.selectbox(
        "Learning rate", ["auto", 10, 50, 100, 200, 500, 1000], index=0
    )
    init = st.selectbox("Init method", ["pca", "random"], index=0)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    run = st.button("Run t-SNE", type="primary")
    if run:
        tsne = TSNE(
            n_components=2,
            perplexity=float(perp),
            # n_iter=int(n_iter),
            learning_rate=lr_choice if lr_choice != "auto" else "auto",
            init=init,
            random_state=int(seed),
            verbose=0,
        )
        emb = tsne.fit_transform(X.values)
        emb_df = pd.DataFrame(emb, index=X.index, columns=["tSNE1", "tSNE2"])

        # 1) ホバーに載せたい列を決定（特徴量 + ラベル（任意））
        hover_cols = [c for c in feats if c in df_used.columns]
        if label_col and label_col != "(none)" and label_col in df_used.columns:
            hover_cols = list(dict.fromkeys(hover_cols + [label_col]))  # 重複除去

        # 2) t-SNE座標に hover用の元データ列を結合（同じindexでjoin）
        plot_df = emb_df.join(df_used.loc[emb_df.index, hover_cols])

        # 3) カラー指定（ラベルがある場合のみ）
        color_arg = (
            label_col
            if (label_col and label_col != "(none)" and label_col in plot_df.columns)
            else None
        )

        # 4) プロット（hover_data は plot_df に存在する列のみを渡す）
        fig = px.scatter(
            plot_df,
            x="tSNE1",
            y="tSNE2",
            color=color_arg,
            hover_data=hover_cols if hover_cols else None,
            template="plotly_white",
            opacity=0.9,
            width=1100,
            height=700,
        )

        # ズーム/パン無効化（現状のままでOK）
        fig.update_layout(dragmode=False)
        config = {
            "scrollZoom": False,
            "doubleClick": False,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "autoScale2d",
                "zoomIn2d",
                "zoomOut2d",
                "resetScale2d",
            ],
        }
        plot_area.plotly_chart(fig, use_container_width=True, config=config)

        st.markdown("### Download")
        st.download_button(
            "Download embedding CSV",
            data=plot_df.to_csv(index=True).encode("utf-8"),
            file_name="tsne_embedding.csv",
            mime="text/csv",
        )
        try:
            import plotly.io as pio

            html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
            st.download_button(
                "Download interactive HTML",
                data=html_str.encode("utf-8"),
                file_name="tsne_plot.html",
                mime="text/html",
            )
        except Exception as e:
            st.info(f"HTML export unavailable: {e}")
