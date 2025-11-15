
import gradio as gr
import torch
from PIL import Image
from clip import clip
import numpy as np

# ---- model ----
device = "mps"
IMG_DTYPE = torch.float16
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.to(dtype=IMG_DTYPE).eval()

TEXT_PROMPTS = ['photograph taken in Poland up to 1944; interwar or older clothing; coats, hats, uniforms; cobblestone streets; prewar tenement houses; horse carriages or very old cars; hand-painted shop signs; art deco typography; sepia or black and white style', "photo from the Polish People's Republic (1945–1989), PRL; prefab panel blocks, RUCH kiosk, neon signs, 'Społem' or 'Pewex' stores; queues and everyday street scenes; 1970s–1980s clothing with shaggy hairstyles, thick-rimmed glasses, polyester suits; Fiat 126p, Polonez, Żuk or Nysa vans; community theater or amateur performances; socialist-era typography and posters", 'photograph in Poland after 1990; modern ads and global brand logos; PVC banners, colorful shop signs; street trade and open markets; modern cars after 2005; sportswear with visible logos; smartphones, glass office buildings, shopping malls, renovated tenement houses']
LABELS = ['do 1944', 'PRL 1945–1989', 'po 1990']
PROMPT_TO_PERIOD = {'photograph taken in Poland up to 1944; interwar or older clothing; coats, hats, uniforms; cobblestone streets; prewar tenement houses; horse carriages or very old cars; hand-painted shop signs; art deco typography; sepia or black and white style': 'do 1944', "photo from the Polish People's Republic (1945–1989), PRL; prefab panel blocks, RUCH kiosk, neon signs, 'Społem' or 'Pewex' stores; queues and everyday street scenes; 1970s–1980s clothing with shaggy hairstyles, thick-rimmed glasses, polyester suits; Fiat 126p, Polonez, Żuk or Nysa vans; community theater or amateur performances; socialist-era typography and posters": 'PRL 1945–1989', 'photograph in Poland after 1990; modern ads and global brand logos; PVC banners, colorful shop signs; street trade and open markets; modern cars after 2005; sportswear with visible logos; smartphones, glass office buildings, shopping malls, renovated tenement houses': 'po 1990'}

def classify_period(image):
    if image is None:
        return "Brak obrazu", {}

    img_tensor = clip_preprocess(image).unsqueeze(0).to(device, dtype=IMG_DTYPE)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        text_tokens = clip.tokenize(TEXT_PROMPTS, truncate=True).to(device)
        text_feat = clip_model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logits = (100.0 * img_feat @ text_feat.T).squeeze(0)
        probs = logits.softmax(dim=-1).cpu().numpy()

    scores = {PROMPT_TO_PERIOD[p]: float(pc * 100) for p, pc in zip(TEXT_PROMPTS, probs)}
    best_period = max(scores, key=scores.get)
    return best_period, scores

demo = gr.Interface(
    fn=classify_period,
    inputs=gr.Image(type="pil", label="Wgraj zdjęcie"),
    outputs=[gr.Label(label="Decyzja"), gr.Label(label="Prawdopodobieństwa")],
    title="Zero-shot: okres historyczny",
)

demo.launch(server_name="127.0.0.1", server_port=7867, show_api=False)
