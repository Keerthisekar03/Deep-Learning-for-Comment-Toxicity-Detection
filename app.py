import sys
import os



# ── 2. IMPORTS ────────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TensorFlow C++ noise

import streamlit as st

try:
    import tensorflow as tf
except ImportError:
    st.error("❌ TensorFlow not found. Run:  pip install tensorflow  (or install to D:\\PythonPackages)")
    st.stop()

import pandas as pd
import plotly.express as px
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# ── 3. PAGE CONFIGURATION ─────────────────────────────────────────────────────
st.set_page_config(page_title="Toxicity Detector", page_icon="🛡️", layout="wide")

# ── 4. MODEL PATH ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Update this line to match the new extension
WEIGHTS_PATH = os.path.join(BASE_DIR, 'toxicity_weights.weights.h5')
VOCAB_PATH   = os.path.join(BASE_DIR, 'vectorizer_vocab.pkl')

# ── 5. MANUAL MODEL REBUILD ──────────────────────────────────────────────────
@st.cache_resource
def load_toxicity_model_manual():
    MAX_FEATURES = 20000
    MAX_LEN = 200
    
    try:
        # 1. Rebuild the Vectorizer layer
        vectorizer = layers.TextVectorization(
            max_tokens=MAX_FEATURES,
            output_sequence_length=MAX_LEN,
            standardize="lower_and_strip_punctuation"
        )
        
        # 2. Load the Vocabulary (Speaking the same "language" as Colab)
        if not os.path.exists(VOCAB_PATH):
            return None, f"❌ Vocabulary file missing at: {VOCAB_PATH}"
            
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        vectorizer.set_vocabulary(vocab)

        # 3. Define the Architecture (The "Skeleton")
        model = models.Sequential([
            layers.Input(shape=(1,), dtype=tf.string),
            vectorizer,
            layers.Embedding(MAX_FEATURES + 1, 128),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2), # Added for better generalization
            layers.Dense(6, activation='sigmoid')
        ])
        
        # 4. Inject the Weights (The "Muscles")
        if not os.path.exists(WEIGHTS_PATH):
            return None, f"❌ Weights file missing at: {WEIGHTS_PATH}"
            
        model.load_weights(WEIGHTS_PATH)
        return model, None

    except Exception as e:
        return None, f"⚠️ Error during manual rebuild: {str(e)}"

# Initialize the model
model, model_error = load_toxicity_model_manual()

# Display status in the sidebar
if model_error:
    st.sidebar.error(model_error)
else:
    st.sidebar.success("✅ Model & Weights loaded successfully!")

# ── 6. CONSTANTS ──────────────────────────────────────────────────────────────
LABELS = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

# ── 7. UI ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Comment Toxicity Detection")
st.subheader("Deep Learning NLP Analysis")

# ── Sidebar: model status ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Status")
    if model_error:
        st.error(f"❌ {model_error}")
    else:
        st.success("✅ Model loaded successfully")
        st.caption(f"Path: {MODEL_PATH}")
    st.divider()
    threshold = st.slider(
        "Flag threshold (%)", min_value=10, max_value=90,
        value=50, step=5,
        help="Scores above this are shown as errors (red), below as success (green)."
    )

st.markdown("---")

# ── Single comment analysis ────────────────────────────────────────────────────
input_text = st.text_area(
    "Enter a comment to analyze:",
    placeholder="Type here…",
    height=150,
)

if st.button("🚀 Predict Toxicity", type="primary", use_container_width=True):
    if model_error or model is None:
        st.error(f"Model is not loaded: {model_error}")
    elif not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing…"):
            try:
                # The model has a built-in TextVectorization layer,
                # so we pass a plain Python list of strings.
                raw = model.predict([input_text], verbose=0)
                scores = raw[0].tolist()          # shape: (6,)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        res_df = pd.DataFrame({'Category': LABELS, 'Confidence Score': scores})

        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.write("### Analysis Results")
            for label, score in zip(LABELS, scores):
                pct = score * 100
                if pct > threshold:
                    st.error(f"**{label}**: {pct:.2f}%")
                elif pct > threshold * 0.4:   # "warning zone" = 40% of threshold
                    st.warning(f"**{label}**: {pct:.2f}%")
                else:
                    st.success(f"**{label}**: {pct:.2f}%")

        with col2:
            fig = px.bar(
                res_df,
                x='Category',
                y='Confidence Score',
                color='Confidence Score',
                color_continuous_scale='Reds',
                range_y=[0, 1],
                text=res_df['Confidence Score'].map(lambda v: f"{v:.1%}"),
                title="Toxicity Confidence Scores",
            )
            fig.add_hline(
                y=threshold / 100,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Threshold ({threshold}%)",
                annotation_position="top left",
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Bulk CSV analysis ──────────────────────────────────────────────────────────
st.divider()
st.write("### 📁 Bulk Upload (CSV)")
st.caption("CSV must have a column named **comment_text**.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'comment_text' not in data.columns:
        st.error("❌ Column `comment_text` not found in the CSV.")
    else:
        st.success(f"✅ Loaded {len(data):,} rows. Click the button below to analyse.")
        st.dataframe(data.head(5), use_container_width=True)

        if st.button("📊 Analyze Full CSV", use_container_width=True):
            if model is None:
                st.error(f"Model not loaded: {model_error}")
            else:
                with st.spinner(f"Analysing {len(data):,} comments…"):
                    try:
                        texts = data['comment_text'].astype(str).tolist()
                        bulk_results = model.predict(texts, verbose=0)  # shape: (N, 6)
                        for i, label in enumerate(LABELS):
                            data[label] = bulk_results[:, i]
                    except Exception as e:
                        st.error(f"Bulk prediction failed: {e}")
                        st.stop()

                st.success("Done!")
                st.dataframe(data.head(20), use_container_width=True)

                csv_bytes = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Full Results",
                    data=csv_bytes,
                    file_name="toxicity_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )