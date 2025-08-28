# app.py

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch

# --- Optional libs for extras ---
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# POS (Unit 2)
import nltk
from nltk import pos_tag, word_tokenize
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Word2Vec baseline (Unit 3)
from gensim.models import Word2Vec

# Optional BioNER (graceful fallback)
from typing import List, Dict, Tuple, Optional
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ---------------------------
# Streamlit Base Config
# ---------------------------
st.set_page_config(page_title="BioMed-QA", page_icon="🧬", layout="wide")

# ---------------------------
# Styles
# ---------------------------
st.markdown("""
<style>
    .metric-container {
        background-color: #111418;
        border: 1px solid #2b2f36;
        padding: 12px;
        border-radius: 8px;
        margin: 6px 0;
    }
    mark.answer { background-color: #FFEB3B; padding: 0.15em 0.3em; border-radius: 0.25em; }
    mark.chem   { background-color: #81C784; padding: 0.15em 0.3em; border-radius: 0.25em; }
    mark.dis    { background-color: #64B5F6; padding: 0.15em 0.3em; border-radius: 0.25em; }
    mark.gene   { background-color: #F06292; padding: 0.15em 0.3em; border-radius: 0.25em; }
    mark.query  { background-color: #B39DDB; padding: 0.15em 0.3em; border-radius: 0.25em; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Fallback Pattern NER
# ---------------------------
class RegexBiomedicalNER:
    def __init__(self):
        disease = [
            r'\b(?:cancer|tumou?r|disease|syndrome|disorder|infection|virus|bacteria|failure)\b',
            r'\b\w+(?:itis|osis|emia|pathy|trophy|plasia|oma|opathy)\b',
            r'\bCOVID-19\b|\bSARS-CoV-2\b|\bAlzheimer\'?s?\b|\bParkinson\'?s?\b'
        ]
        chemical = [
            r'\b\w+(?:ine|ate|ide|ase|ol|al|ic acid|amine|azole|azide|cycline|mycin|cillin)\b',
            r'\b(?:drug|medication|compound|inhibitor|agonist|antagonist)\b',
            r'\b(?:aspirin|ibuprofen|insulin|penicillin|morphine|caffeine|acetaminophen|carbamazepine|lamotrigine)\b'
        ]
        gene = [
            r'\b[A-Z0-9]{2,6}\d?\b',                                 # e.g., TP53, BRCA1
            r'\b(?:gene|protein|receptor|kinase|enzyme)\b',
            r'\b\w+(?:-receptor| gene)\b'
        ]
        self.rx_dis = re.compile('|'.join(disease), re.IGNORECASE)
        self.rx_chem = re.compile('|'.join(chemical), re.IGNORECASE)
        self.rx_gene = re.compile('|'.join(gene), re.IGNORECASE)

    def extract(self, text: str) -> List[Dict]:
        ents = []
        for m in self.rx_dis.finditer(text):
            ents.append({"text": m.group(), "label": "DISEASE", "start": m.start(), "end": m.end()})
        for m in self.rx_chem.finditer(text):
            ents.append({"text": m.group(), "label": "CHEMICAL", "start": m.start(), "end": m.end()})
        for m in self.rx_gene.finditer(text):
            ents.append({"text": m.group(), "label": "GENE", "start": m.start(), "end": m.end()})
        # dedupe by (text,label,start)
        seen = set()
        out = []
        for e in sorted(ents, key=lambda x: x["start"]):
            key = (e["text"].lower(), e["label"], e["start"])
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_resource
def load_data_and_models():
    # 1) Data
    try:
        df = pd.read_csv("pubmed_sample_processed.csv")
    except FileNotFoundError:
        st.error("Required file `pubmed_sample_processed.csv` not found. Run `python preprocess.py` first.")
        st.stop()

    # 2) Dense embedder
    embedder = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    # 3) Corpus embeddings
    try:
        with open('corpus_embeddings.pkl', 'rb') as f:
            corpus_embeddings = pickle.load(f)
        if not torch.is_tensor(corpus_embeddings):
            corpus_embeddings = torch.tensor(corpus_embeddings, dtype=torch.float32)
    except FileNotFoundError:
        st.error("`corpus_embeddings.pkl` not found. Run `python preprocess.py` first.")
        st.stop()

    # 4) TF-IDF (fit on our processed df)
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5,
                            ngram_range=(1, 2), max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df['abstractText'])

    return df, embedder, corpus_embeddings, tfidf, tfidf_matrix

@st.cache_resource
def load_word2vec(df: pd.DataFrame, size: int = 100):
    # Train a quick W2V on abstracts (cached)
    sentences = [word_tokenize(t.lower()) for t in df['abstractText'].astype(str).tolist()]
    model = Word2Vec(sentences, vector_size=size, window=5, min_count=3, workers=2, epochs=8)
    return model

@st.cache_resource
def load_biobert_ner_pipeline(model_id: str = "d4data/biomedical-ner-all"):
    """
    Tries to create a HF token-classification pipeline. If anything fails,
    return None and we'll fall back to regex NER.
    """
    if not HF_AVAILABLE:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        m = AutoModelForTokenClassification.from_pretrained(model_id)
        return pipeline("token-classification", model=m, tokenizer=tok, aggregation_strategy="simple")
    except Exception:
        return None

# ---------------------------
# Retrieval functions
# ---------------------------
def search_dense(query: str, embedder, corpus_embeddings, df: pd.DataFrame, top_k: int):
    q = embedder.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q, corpus_embeddings)[0]
    topk = torch.topk(scores, k=min(top_k, len(scores)))
    idx = topk.indices.tolist()
    vals = topk.values.tolist()
    res = df.iloc[idx].copy()
    res["score"] = vals
    res["rank"] = np.arange(1, len(res) + 1)
    return res

def search_tfidf(query: str, vectorizer, tfidf_matrix, df: pd.DataFrame, top_k: int):
    v = vectorizer.transform([query])
    sims = cosine_similarity(v, tfidf_matrix).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    res = df.iloc[idx].copy()
    res["score"] = sims[idx]
    res["rank"] = np.arange(1, len(res) + 1)
    return res

def embed_doc_w2v(text: str, w2v: Word2Vec, size: int = 100):
    tokens = [t for t in word_tokenize(text.lower()) if t in w2v.wv.key_to_index]
    if not tokens:
        return np.zeros(size, dtype=np.float32)
    return np.mean([w2v.wv[t] for t in tokens], axis=0)

@st.cache_data(show_spinner=False)
def build_w2v_doc_matrix(df: pd.DataFrame, _w2v, size: int = 100):
    mat = np.vstack([embed_doc_w2v(t, _w2v, size) for t in df['abstractText'].astype(str)])
    return mat

def search_w2v(query: str, w2v: Word2Vec, doc_matrix: np.ndarray, df: pd.DataFrame, top_k: int):
    qv = embed_doc_w2v(query, w2v, w2v.vector_size)
    sims = cosine_similarity([qv], doc_matrix).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    res = df.iloc[idx].copy()
    res["score"] = sims[idx]
    res["rank"] = np.arange(1, len(res) + 1)
    return res

def search_hybrid(query, embedder, corpus_embeddings, tfidf, tfidf_matrix, df, top_k, alpha=0.7):
    dense = search_dense(query, embedder, corpus_embeddings, df, len(df))
    tf = search_tfidf(query, tfidf, tfidf_matrix, df, len(df))
    d_scores = (dense["score"] / dense["score"].max()).to_dict()
    t_scores = (tf["score"] / tf["score"].max()).to_dict()
    combined = {}
    for i in df.index:
        combined[i] = alpha * d_scores.get(i, 0) + (1 - alpha) * t_scores.get(i, 0)
    top_idx = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]
    res = df.loc[top_idx].copy()
    res["score"] = [combined[i] for i in top_idx]
    res["rank"] = np.arange(1, len(res) + 1)
    return res

# ---------------------------
# NER utilities
# ---------------------------
def run_regex_ner(text: str, rx: RegexBiomedicalNER):
    return rx.extract(text)

def run_bio_ner(text: str, pipe):
    out = []
    spans = pipe(text)
    for sp in spans:
        label = sp["entity_group"].upper()
        if "CHEM" in label: label = "CHEMICAL"
        if "DISEASE" in label or label == "DIS": label = "DISEASE"
        if "GENE" in label or "PROTEIN" in label: label = "GENE"
        out.append({"text": sp["word"], "label": label, "start": sp["start"], "end": sp["end"]})
    return out

def color_for_label(label: str) -> str:
    return {"CHEMICAL": "chem", "DISEASE": "dis", "GENE": "gene"}.get(label, "answer")

def highlight(text: str, entities: List[Dict], extra_keywords: Optional[List[str]] = None):
    # Sort spans left->right to safely inject marks
    spans = sorted(entities, key=lambda e: e["start"])
    offset = 0
    s = text
    for e in spans:
        klass = color_for_label(e["label"])
        start, end = e["start"] + offset, e["end"] + offset
        frag = s[start:end]
        ins = f"<mark class='{klass}'>{frag}</mark>"
        s = s[:start] + ins + s[end:]
        offset += len(ins) - (end - start)
    # Highlight query keywords (after entity coloring)
    if extra_keywords:
        for kw in sorted(set(k for k in extra_keywords if len(k) > 3), key=len, reverse=True):
            s = re.sub(rf"(?i)\\b{re.escape(kw)}\\b", f"<mark class='query'>{kw}</mark>", s)
    return s

# ---------------------------
# IR metrics (interpretable)
# ---------------------------
def precision_at_k(rels: List[int], k=5):
    r = rels[:k]
    return np.sum(r) / max(1, len(r))

def recall_at_k(rels: List[int], k=5):
    total_rel = np.sum(rels)
    if total_rel == 0: return 0.0
    r = rels[:k]
    return np.sum(r) / total_rel

def dcg_at_k(rels: List[int], k=5):
    rels = np.array(rels[:k], dtype=float)
    if len(rels) == 0: return 0.0
    discounts = 1 / np.log2(np.arange(2, len(rels) + 2))
    return np.sum(rels * discounts)

def ndcg_at_k(rels: List[int], k=5):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return 0.0 if ideal == 0 else dcg / ideal

def evaluate_ir_for_query(query: str, rel_keywords: List[str],
                          run_fn, k=5) -> Dict[str, float]:
    res = run_fn(query)
    # simple heuristic relevance: an abstract is relevant if it contains any keyword
    rels = []
    for _, row in res.iterrows():
        txt = row["abstractText"].lower()
        rels.append(1 if any(kw.lower() in txt for kw in rel_keywords) else 0)
    return {
        "P@k": precision_at_k(rels, k),
        "R@k": recall_at_k(rels, k),
        "nDCG@k": ndcg_at_k(rels, k)
    }

# ---------------------------
# App Body
# ---------------------------
st.title("🧬 BioMed-QA: Advanced Biomedical Question Answering")
st.caption("Sequence Labeling & Semantics • Vector Semantics & Embeddings • IR & QA")

with st.spinner("Loading models & data…"):
    df, embedder, corpus_embeddings, tfidf, tfidf_matrix = load_data_and_models()
st.success("Models and data loaded successfully!")

# Optional components
regex_ner = RegexBiomedicalNER()
biobert_pipe = load_biobert_ner_pipeline()  # None if not available

# Word2Vec baseline
with st.spinner("Preparing Word2Vec baseline (first run may take ~20s)…"):
    w2v = load_word2vec(df, size=100)
    w2v_doc_matrix = build_w2v_doc_matrix(df, w2v, size=100)

# Header metrics
c1, c2, c3 = st.columns(3)
c1.metric("📚 Total Abstracts", f"{len(df)}")
c2.metric("🔤 TF-IDF Vocabulary", f"{len(tfidf.vocabulary_)}")
c3.metric("🧪 NER", "BioNER" if biobert_pipe else "Regex (baseline)")

# Tabs
tab_qa, tab_ir, tab_analytics, tab_ner_eval, tab_pos_wsd = st.tabs([
    "❓ QA System", "🔍 IR Comparison", "📊 Analytics", "🧪 NER Evaluation", "🔤 POS & WSD"
])

# ---------------------------
# TAB 1: QA
# ---------------------------
with tab_qa:
    st.header("Biomedical QA System")

    # --- Sidebar option for title preference ---
    with st.sidebar:
        st.subheader("🔖 Document Title Settings")
        title_pref = st.selectbox(
            "Preferred Title Field",
            ["Title", "pmid"],
            index=0,
            help="Choose whether to show paper Title or PubMed ID."
        )

    examples = [
        "What chemicals are used to treat Parkinson's disease?",
        "Which genes are associated with breast cancer?",
        "What drugs can treat COVID-19 symptoms?",
        "Which proteins are involved in Alzheimer's disease?",
        "What compounds are effective against viral infections?"
    ]
    ex = st.selectbox("Choose an example:", ["—"]+examples)
    custom_q = st.text_area("Or type your own question:")
    user_q = custom_q if custom_q.strip() else ("" if ex=="—" else ex)

    colA, colB = st.columns(2)
    with colA:
        search_method = st.radio("Search method:",
                                 ["Hybrid", "Dense (BioBERT)", "TF-IDF", "Word2Vec"],
                                 index=0, horizontal=True)
    with colB:
        k_docs = st.slider("Number of documents:", min_value=5, max_value=20, value=10, step=1)

    alpha = 0.7
    if search_method == "Hybrid":
        alpha = st.slider("Hybrid weight (α): Dense ↔ TF-IDF", 0.0, 1.0, 0.7, 0.05,
                          help="α=1.0 uses only Dense; α=0 uses only TF-IDF")

    # --- Simplified NER options ---
    ner_choice = st.selectbox("Answer extractor (NER):",
                              ["BioNER (HuggingFace)", "Regex baseline"])
    if ner_choice == "BioNER (HuggingFace)" and biobert_pipe is None:
        st.warning("BioNER model not available/offline. Falling back to Regex.")
        ner_choice = "Regex baseline"

    go_btn = st.button("🔎 Search")

    def do_search(q):
        if search_method == "Dense (BioBERT)":
            return search_dense(q, embedder, corpus_embeddings, df, k_docs)
        elif search_method == "TF-IDF":
            return search_tfidf(q, tfidf, tfidf_matrix, df, k_docs)
        elif search_method == "Word2Vec":
            return search_w2v(q, w2v, w2v_doc_matrix, df, k_docs)
        else:
            return search_hybrid(q, embedder, corpus_embeddings, tfidf, tfidf_matrix, df, k_docs, alpha)

    if go_btn:
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            results = do_search(user_q)

            # NER on retrieved docs
            all_answers = []
            highlighted = []
            q_tokens = re.findall(r"\b\w+\b", user_q.lower())

            for idx, row in results.iterrows():
                text = str(row["abstractText"])

                # --- Title Handling with preference + fallbacks ---
                title = None
                if title_pref == "Title" and "Title" in row and pd.notna(row["Title"]) and str(row["Title"]).strip():
                    if "pmid" in row and pd.notna(row["pmid"]):
                        title = f"{str(row['Title']).strip()} (PMID: {int(row['pmid'])})"
                    else:
                        title = str(row["Title"]).strip()
                elif title_pref == "pmid" and "pmid" in row and pd.notna(row["pmid"]):
                    title = f"PubMed Abstract #{int(row['pmid'])}"
                else:
                    title = f"PubMed Abstract #{idx}"

                # --- Run NER ---
                if ner_choice == "Regex baseline":
                    ents = run_regex_ner(text, regex_ner)
                else:
                    ents = run_bio_ner(text, biobert_pipe)

                # Simple context filter: keep spans in sentences that mention some query token
                keep = []
                for e in ents:
                    L = max(0, e["start"] - 120)
                    R = min(len(text), e["end"] + 120)
                    sent = text[L:R].lower()
                    if any(tok in sent for tok in q_tokens if len(tok) > 3):
                        keep.append(e)

                if keep:
                    all_answers.extend([e["text"] for e in keep])

                highlighted_text = highlight(text, keep, q_tokens)
                highlighted.append((title, highlighted_text, float(row["score"])))

            # Answers
            st.subheader("Answers")
            if all_answers:
                uniq = sorted(set(all_answers), key=str.lower)
                st.markdown(" ".join([f"<mark class='answer'>{a}</mark>" for a in uniq]), unsafe_allow_html=True)
                st.info(f"Found {len(uniq)} unique candidate entities.")
            else:
                st.write("No high-confidence entities found in retrieved documents.")

            # Docs
            st.subheader("Documents")
            for i, (title, html_text, score) in enumerate(highlighted, 1):
                with st.expander(f"{title}  (Score {score:.3f})", expanded=False):
                    st.markdown(html_text, unsafe_allow_html=True)


# ---------------------------
# TAB 2: IR Comparison + Metrics
# ---------------------------
with tab_ir:
    st.header("Compare IR Methods")

    comp_examples = ["Kidney Failure", "Parkinson's treatment", "Alzheimer's disease"]
    ir_q = st.selectbox("Query for comparison:", comp_examples)
    custom_ir = st.text_input("Or custom query:")
    if custom_ir.strip(): ir_q = custom_ir

    k = 5  # 👈 define k for evaluation metrics

    if ir_q.strip():
        tf_res = search_tfidf(ir_q, tfidf, tfidf_matrix, df, k)
        de_res = search_dense(ir_q, embedder, corpus_embeddings, df, k)
        hy_res = search_hybrid(ir_q, embedder, corpus_embeddings, tfidf, tfidf_matrix, df, k)

        # Score comparison (bars)
        fig = make_subplots(rows=1, cols=3, subplot_titles=("TF-IDF Scores", "Dense Scores", "Hybrid Scores"))
        fig.add_trace(go.Bar(x=[str(i) for i in tf_res["rank"]], y=tf_res["score"]), 1, 1)
        fig.add_trace(go.Bar(x=[str(i) for i in de_res["rank"]], y=de_res["score"]), 1, 2)
        fig.add_trace(go.Bar(x=[str(i) for i in hy_res["rank"]], y=hy_res["score"]), 1, 3)
        fig.update_layout(height=360, showlegend=False, title=f"Retrieval Comparison for: '{ir_q}'")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### IR Evaluation Metrics (interpretable)")
        st.caption("We consider a result **relevant** if its abstract contains any of the query words (simple heuristic).")

        # Build relevance lists for each method
        q_tokens = [w for w in re.findall(r"\b\w+\b", ir_q.lower()) if len(w) > 3]
        def rels_from(dfres):
            rels = []
            for _, r in dfres.iterrows():
                txt = r["abstractText"].lower()
                rels.append(1 if any(w in txt for w in q_tokens) else 0)
            return rels

        rel_tf = rels_from(tf_res)
        rel_de = rels_from(de_res)
        rel_hy = rels_from(hy_res)

        def summary(name, rels, k=5):
            return {
                "Method": name,
                "P@k": precision_at_k(rels, k),
                "R@k": recall_at_k(rels, k),
                "nDCG@k": ndcg_at_k(rels, k)
            }

        df_metrics = pd.DataFrame([
            summary("TF-IDF", rel_tf, k),
            summary("Dense", rel_de, k),
            summary("Hybrid", rel_hy, k),
        ])

        # Pretty bars
        col_m1, col_m2, col_m3 = st.columns(3)
        for metric, col in zip(["P@k", "R@k", "nDCG@k"], [col_m1, col_m2, col_m3]):
            figm = px.bar(df_metrics, x="Method", y=metric, range_y=[0, 1], text=metric)
            figm.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            figm.update_layout(height=280, title=metric, uniformtext_minsize=12, uniformtext_mode='hide', margin=dict(t=40))
            col.plotly_chart(figm, use_container_width=True)

        st.dataframe(df_metrics.style.format({c: "{:.2f}" for c in ["P@k", "R@k", "nDCG@k"]}),
                     use_container_width=True)
        st.info("**How to read:**\n- **P@k**: fraction of the top-k that are relevant\n- **R@k**: fraction of all relevant items that appear in the top-k\n- **nDCG@k**: quality of ranking—higher if relevant items appear higher up.")

# ---------------------------
# TAB 3: Analytics
# ---------------------------
with tab_analytics:
    st.header("Dataset Analytics")

    colA, colB = st.columns(2)
    with colA:
        lens = df['abstractText'].astype(str).str.len()
        h = px.histogram(lens, nbins=60, title="Abstract Lengths")
        h.update_xaxes(title="Characters")
        h.update_yaxes(title="Count")
        st.plotly_chart(h, use_container_width=True)

    with colB:
        # Top TF-IDF terms table
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        topN = 30
        top_terms = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:topN]
        df_terms = pd.DataFrame(top_terms, columns=["Term", "TF-IDF Score"])
        st.subheader("Top TF-IDF Terms")
        st.dataframe(df_terms, use_container_width=True)

# ---------------------------
# TAB 4: NER Evaluation (tiny demo)
# ---------------------------
with tab_ner_eval:
    st.header("NER Evaluation")
    st.caption("Compares Regex baseline vs BioNER (if available) on a tiny built-in gold sample. Replace this with your own labeled set for stronger results.")

    # A very small toy gold set (text -> entities)
    gold = [
        {
            "text": "Carbamazepine is a drug sometimes used for focal seizures in epilepsy.",
            "CHEMICAL": ["Carbamazepine"],
            "DISEASE": ["epilepsy"],
            "GENE": []
        },
        {
            "text": "Mutations in BRCA1 increase risk of breast cancer.",
            "CHEMICAL": [],
            "DISEASE": ["cancer"],
            "GENE": ["BRCA1"]
        }
    ]

    def evaluate_ner(run_fn):
        tp = {"CHEMICAL":0,"DISEASE":0,"GENE":0}
        fp = {"CHEMICAL":0,"DISEASE":0,"GENE":0}
        fn = {"CHEMICAL":0,"DISEASE":0,"GENE":0}
        for item in gold:
            text = item["text"]
            pred = run_fn(text)
            pred_map = {"CHEMICAL": set(), "DISEASE": set(), "GENE": set()}
            for p in pred:
                pred_map.setdefault(p["label"], set()).add(p["text"].lower())
            for lab in ["CHEMICAL","DISEASE","GENE"]:
                gold_set = set([x.lower() for x in item[lab]])
                pred_set = pred_map.get(lab, set())
                tp[lab] += len(gold_set & pred_set)
                fp[lab] += len(pred_set - gold_set)
                fn[lab] += len(gold_set - pred_set)
        rows = []
        for lab in ["CHEMICAL","DISEASE","GENE"]:
            P = tp[lab] / max(1, tp[lab] + fp[lab])
            R = tp[lab] / max(1, tp[lab] + fn[lab])
            F1 = 0.0 if (P+R)==0 else 2*P*R/(P+R)
            rows.append([lab, tp[lab], fp[lab], fn[lab], P, R, F1])
        return pd.DataFrame(rows, columns=["Label","TP","FP","FN","Precision","Recall","F1"])

    rx_df = evaluate_ner(lambda t: run_regex_ner(t, regex_ner))
    st.subheader("Regex Baseline")
    st.dataframe(rx_df.style.format({"Precision":"{:.2f}","Recall":"{:.2f}","F1":"{:.2f}"}), use_container_width=True)

    if biobert_pipe:
        bio_df = evaluate_ner(lambda t: run_bio_ner(t, biobert_pipe))
        st.subheader("BioNER (Hugging Face)")
        st.dataframe(bio_df.style.format({"Precision":"{:.2f}","Recall":"{:.2f}","F1":"{:.2f}"}), use_container_width=True)
    else:
        st.info("BioNER model not available; using Regex only. (Optional: connect to internet and restart to enable BioNER.)")

# ---------------------------
# TAB 5: POS & WSD
# ---------------------------
with tab_pos_wsd:
    st.header("POS Tagging & Word Sense / Semantics")

    sample_text = st.text_area("Try POS tagging on any sentence:", value="Aspirin is used to treat cardiovascular diseases.")
    if sample_text.strip():
        try:
            tokens = word_tokenize(sample_text)
            tags = pos_tag(tokens)
            df_pos = pd.DataFrame(tags, columns=["Token","POS"])
            st.dataframe(df_pos, use_container_width=True, height=220)
        except Exception as e:
            st.warning(f"POS tagging unavailable: {e}")

    st.subheader("Word Sense & Semantic Similarity (WSD intuition)")
    st.caption("We use cosine similarity in embedding space to disambiguate meanings. The same word can mean different things; context moves it toward the right region in space.")
    pairs = ["cold virus | cold weather", "mouse protein | mouse click", "bank river | bank money"]
    wsd_q = st.selectbox("Choose pair:", pairs)
    custom_pair = st.text_input("Or enter custom pair (use |):")
    if "|" in custom_pair: wsd_q = custom_pair

    if "|" in wsd_q:
        a, b = [x.strip() for x in wsd_q.split("|", 1)]
        va = embedder.encode(a, convert_to_tensor=True)
        vb = embedder.encode(b, convert_to_tensor=True)
        sim = float(util.cos_sim(va, vb)[0][0])
        st.metric("Cosine similarity", f"{sim:.3f}",
                  help="Closer to 1.0 means more similar meaning; near 0 means unrelated.")


# ---------------------------
# Footer / Methodology block
# ---------------------------
#with st.expander("🔬 Methodology (for your report)"):
 #   st.markdown("""
#**Architecture:**  
#Query → Retrieval (TF-IDF / Dense BioBERT / Word2Vec / Hybrid) → Top-k abstracts → NER (Regex or BioNER) → Highlighted answers.

#**Unit 2:** POS Tagging demo (NLTK). NER with Regex baseline + optional BioNER. WSD intuition via cosine similarity.  
#**Unit 3:** TF-IDF baseline + Word2Vec baseline + Dense embeddings (BioBERT).  
#**Unit 4:** IR system with metrics (P@k, R@k, nDCG@k) and score visualizations.
#""")
