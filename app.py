import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Constants / regex
# ============================================================
TEAM_CODES = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
}
TEAM_RE = re.compile(r"\b(" + "|".join(sorted(TEAM_CODES)) + r")\b", flags=re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
MULTISPACE_RE = re.compile(r"\s+")


# ============================================================
# Text normalization 
# ============================================================
def normalize_text(s: str) -> str:
    """
    Normalize user query text so the model learns intent structure,
    not memorized entities.
    """
    s = str(s).strip().lower()

    # normalize separators
    s = re.sub(r"[_/\\|]+", " ", s)
    s = re.sub(r"[()\[\]{}]", " ", s)

    # mask years and team abbreviations
    s = YEAR_RE.sub(" YEAR ", s)
    s = TEAM_RE.sub(" TEAM ", s)

    # conservative shorthand expansions
    s = s.replace("yds", " yards")
    s = s.replace("tds", " touchdowns")

    # strip punctuation but keep ? and %
    s = re.sub(r"[^\w\s\?%]+", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


# ============================================================
# Data + training helpers
# ============================================================
def load_dataset(df: pd.DataFrame, group_col: Optional[str]) -> tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    required = {"Question", "Intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Intent"] = df["Intent"].astype(str).str.strip()
    df = df[(df["Question"] != "") & (df["Intent"] != "")].copy()

    if len(df) < 50:
        raise ValueError(f"Dataset too small ({len(df)} rows). Add more examples per intent.")

    groups = None
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found. Columns: {list(df.columns)}")
        groups = df[group_col].astype(str).replace("", np.nan)
        if groups.isna().any():
            raise ValueError(f"Group column '{group_col}' contains empty values; fill all Canonical ids.")
    return df["Question"], df["Intent"], groups


def build_pipeline(min_df: int, char_ngram_min: int, char_ngram_max: int, C: float, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(char_ngram_min, char_ngram_max),
                    min_df=min_df,
                    strip_accents="unicode",
                    lowercase=False,
                    preprocessor=normalize_text,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=max_iter,
                    C=C,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def grouped_split(X: pd.Series, y: np.ndarray, groups: pd.Series, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[tr_idx], X.iloc[va_idx], y[tr_idx], y[va_idx]


def pretty_confusion(cm: np.ndarray, class_names: list[str]) -> str:
    header = " " * 18 + " ".join([f"{c[:14]:>14}" for c in class_names])
    lines = ["Confusion Matrix (rows=true, cols=pred):", header]
    for i, row in enumerate(cm):
        label = class_names[i][:14]
        lines.append(f"{label:>14} (true) " + " ".join([f"{int(v):>14d}" for v in row]))
    return "\n".join(lines)


# ============================================================
# Model
# ============================================================
def model_paths(outdir: str) -> tuple[Path, Path]:
    base = Path(outdir).expanduser()
    return base / "intents_pipeline.joblib", base / "label_encoder.joblib"


def save_artifacts(outdir: str, pipeline: Pipeline, le: LabelEncoder) -> tuple[Path, Path]:
    base = Path(outdir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    p_model, p_le = model_paths(outdir)
    joblib.dump(pipeline, p_model)
    joblib.dump(le, p_le)
    return p_model, p_le


def load_artifacts(outdir: str) -> tuple[Optional[Pipeline], Optional[LabelEncoder]]:
    p_model, p_le = model_paths(outdir)
    if p_model.exists() and p_le.exists():
        return joblib.load(p_model), joblib.load(p_le)
    return None, None


def predict_intent(pipeline: Pipeline, le: LabelEncoder, text: str):
    probs = pipeline.predict_proba([text])[0]
    best_idx = int(np.argmax(probs))
    best_p = float(probs[best_idx])
    best_label = le.inverse_transform([best_idx])[0]
    topk = np.argsort(probs)[::-1][:3]
    top3 = [(le.inverse_transform([int(i)])[0], float(probs[int(i)])) for i in topk]
    return best_label, best_p, top3


def friendly_intent_line(intent: str, conf: float) -> str:
    pct = conf * 100.0
    # Simple, human-friendly phrasing (adjust to match your 7-intent set if you want)
    return f"I think you’re asking about **{intent}** (confidence: **{pct:.1f}%**)."


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="NFL Stats Chatbot", page_icon="🏈", layout="centered")
st.title("NFL Statistics Chatbot")
st.caption("Retrieval Chatbot for NFL stats questions")

with st.sidebar:
    st.header("Settings")

    outdir = st.text_input("Artifacts directory", value="model")
    st.divider()

    st.subheader("Training")
    group_col = st.text_input("Group column (optional, recommended)", value="Canonical")
    test_size = st.slider("Validation split size", 0.10, 0.40, 0.20, 0.05)
    seed = st.number_input("Random seed", value=42, step=1)

    st.subheader("Vectorizer / model")
    min_df = st.slider("TF-IDF min_df", 1, 10, 2, 1)
    char_min = st.slider("Char n-gram min", 2, 6, 3, 1)
    char_max = st.slider("Char n-gram max", 3, 8, 5, 1)
    C = st.slider("LogReg C", 0.1, 5.0, 2.0, 0.1)
    max_iter = st.slider("LogReg max_iter", 500, 6000, 3000, 500)

    st.divider()
    show_debug = st.checkbox("Show debug panel (top-3)", value=False)


tab_train, tab_chat = st.tabs(["Train", "Chat"])


# ----------------------------
# Train tab
# ----------------------------
with tab_train:
    st.subheader("Train the intent classifier")
    st.write("Upload a CSV with columns: Question, Intent,Canonical).")

    uploaded = st.file_uploader("Training CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("Train model", type="primary"):
            try:
                X_text, y_labels, groups = load_dataset(df, group_col.strip() or None)

                le = LabelEncoder()
                y = le.fit_transform(y_labels)

                if groups is not None:
                    X_train, X_val, y_train, y_val = grouped_split(X_text, y, groups, test_size, int(seed))
                    split_note = f"Grouped split by '{group_col}'"
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_text,
                        y,
                        test_size=test_size,
                        random_state=int(seed),
                        stratify=y,
                    )
                    split_note = "Stratified random split (may leak paraphrases)"

                pipeline = build_pipeline(min_df, char_min, char_max, float(C), int(max_iter))
                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_val)
                acc = accuracy_score(y_val, y_pred)

                st.success("Training complete.")
                st.write(f"Split: **{split_note}**")
                st.write(f"Examples: train={len(X_train)}  val={len(X_val)}")
                st.write(f"Accuracy: **{acc:.4f}**")

                labels_present = np.unique(y_val)
                report = classification_report(
                    y_val,
                    y_pred,
                    labels=labels_present,
                    target_names=le.inverse_transform(labels_present),
                    digits=4,
                    zero_division=0,
                )
                st.text("Classification Report:\n" + report)

                cm = confusion_matrix(y_val, y_pred, labels=np.arange(len(le.classes_)))
                st.text(pretty_confusion(cm, list(le.classes_)))

                p_model, p_le = save_artifacts(outdir, pipeline, le)
                st.info(f"Saved artifacts:\n- {p_model}\n- {p_le}")

                st.session_state["pipeline"] = pipeline
                st.session_state["label_encoder"] = le

            except Exception as e:
                st.error(f"Training failed: {e}")


# ----------------------------
# Chat tab
# ----------------------------
with tab_chat:
    st.subheader("Chat")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Load model from session or disk
    pipeline = st.session_state.get("pipeline")
    le = st.session_state.get("label_encoder")

    if pipeline is None or le is None:
        pipeline, le = load_artifacts(outdir)
        if pipeline and le:
            st.session_state["pipeline"] = pipeline
            st.session_state["label_encoder"] = le
            st.caption(f"Loaded artifacts from `{outdir}/`.")
        else:
            st.warning("No model loaded yet. Train a model in the **Train** tab first.")
            st.stop()

    # Render history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def bot_say(text: str):
        st.session_state["messages"].append({"role": "assistant", "content": text})
        with st.chat_message("assistant"):
            st.markdown(text)

    user_text = st.chat_input("Ask an NFL stats question…")

    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        intent, conf, top3 = predict_intent(pipeline, le, user_text)
        bot_say(friendly_intent_line(intent, conf))

        if show_debug:
            with st.expander("Debug", expanded=False):
                st.write({"intent": intent, "confidence": conf})
                st.write({"top3": top3})
