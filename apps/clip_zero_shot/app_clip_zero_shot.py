import streamlit as st
from PIL import Image
import torch
import clip
import numpy as np

# ==========================
#       MODEL CLIP
# ==========================

device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ==========================
#     PROMPTY / LABELS
# ==========================

LABELS = ["do 1944", "PRL 1945–1989", "po 1990"]

TEXT_PROMPTS = [
    "Polish photograph before 1944. Old cars, prewar clothing, tenement houses, sepia or BW.",
    "Polish People's Republic (1945–1989). PRL symbols: prefabs, RUCH kiosk, neon signs, Polonez, Fiat 126p.",
    "Poland after 1990. Modern ads, smartphones, renovated buildings, modern cars."
]

# ==========================
#     FUNKCJA ZERO-SHOT
# ==========================

@torch.no_grad()
def classify_period(img: Image.Image):
    img_input = clip_preprocess(img).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(TEXT_PROMPTS).to(device)

    img_features = clip_model.encode_image(img_input)
    img_features /= img_features.norm(dim=-1, keepdim=True)

    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits = (img_features @ text_features.T)[0]
    probs = logits.softmax(dim=-1).cpu().numpy()

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


# ==========================
#          STYL CSS
# ==========================

st.set_page_config(page_title="Zero-shot okresów historycznych", layout="wide")

st.markdown("""
<style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1 {
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.2rem !important;
    }

    .result-box {
        background-color: #fafafa;
        padding: 20px 25px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    .decision-box {
        background-color: #e9f7ef;
        padding: 16px 20px;
        border-radius: 10px;
        border: 1px solid #b6e1c5;
    }

    .bar-container {
        height: 10px;
        background-color: #eee;
        border-radius: 5px;
        margin-top: 4px;
        margin-bottom: 12px;
        position: relative;
    }

    .bar-fill {
        height: 10px;
        background-color: #ffcc80;
        border-radius: 5px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================
#           UI
# ==========================

st.title("Zero-shot: przed 1945 vs PRL vs po 1990")

uploaded = st.file_uploader("Wgraj zdjęcie", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # --- KOLUMNY: OBRAZ | WYNIKI ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.image(img, caption="Wgrany obraz", use_column_width=True)

    with col_right:
        res = classify_period(img)
        best = max(res, key=res.get)

        # --- Wyniki w ramce ---
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("Prawdopodobieństwa:")

        for label, value in res.items():
            pct = value * 100
            st.markdown(f"**{label}:** {pct:.1f}%")

            bar_html = f"""
            <div class='bar-container'>
                <div class='bar-fill' style='width:{pct}%;'></div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # --- Decyzja ---
        st.markdown("<div class='decision-box'>", unsafe_allow_html=True)
        st.subheader("Decyzja:")
        st.write(f"<b>{best}</b>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)