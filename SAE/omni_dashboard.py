import os, sys, torch, random, gc, json
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
import plotly.graph_objects as go
from collections import Counter

sys.path.append(os.path.expanduser("~/Projects")) 
from dictionary_learning.trainers.top_k import AutoEncoderTopK

BASE_DIR   = "/home/adam/Projects"
IMAGE_DIR  = os.path.join(BASE_DIR, "data/test2014")
SAE_DIR    = os.path.join(BASE_DIR, "trained_sae")
DASHBOARD_HTML = os.path.join(BASE_DIR, "neural_flow_dashboard.html")
CSV_PATH   = os.path.join(BASE_DIR, "feature_paths_omni.csv")

DEVICE      = "cuda:0" 
LAYERS      = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
BATCH_SIZE  = 16 
SAMPLE_SIZE = 1280
METHODS     = ["max", "sum", "tfidf"]
IMG_SIZE, COLS, ROWS, HEADER_H = 128, 10, 10, 60   

DIRS, JSON_PATHS, JSON_DATA = {}, {}, {}
poster_features = {m: {L: [] for L in LAYERS} for m in METHODS}

for m in METHODS:
    d = os.path.join(BASE_DIR, f"layer_galleries_{m}")
    os.makedirs(d, exist_ok=True)
    DIRS[m] = d
    JSON_PATHS[m] = os.path.join(d, f"gallery_data_{m}.json")
    JSON_DATA[m] = {"dataset_info": {"total": SAMPLE_SIZE}, "layers": {}}

jsons_exist = all(os.path.exists(JSON_PATHS[m]) for m in METHODS)
csv_exists = os.path.exists(CSV_PATH)

if csv_exists and jsons_exist:
    df = pd.read_csv(CSV_PATH)
    
    for m in METHODS:
        with open(JSON_PATHS[m], "r") as f:
            g_data = json.load(f)
        for L in LAYERS:
            layer_key = f"layer_{L}"
            if layer_key in g_data["layers"]:
                poster_features[m][L] = [int(k.split("_")[1]) for k in g_data["layers"][layer_key].keys()]
else:
    vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).half()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    random.seed(42)
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    sampled_images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))
    sankey_data = {img: {"image": img} for img in sampled_images}

    def save_poster(method, layer, top_feats, score_matrix, raw_acts):
        canvas = Image.new('RGB', (COLS * IMG_SIZE, HEADER_H + (ROWS * IMG_SIZE)), color=(30, 30, 30))
        draw = ImageDraw.Draw(canvas)
        try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except: font = ImageFont.load_default()

        poster_features[method][layer] = [f.item() for f in top_feats]
        layer_key = f"layer_{layer}"
        JSON_DATA[method][layer_key] = {}

        for col_idx, f_idx in enumerate(top_feats):
            f_idx = f_idx.item()
            JSON_DATA[method][layer_key][f"feature_{f_idx}"] = [] 
            
            valid_mask = raw_acts[:, f_idx] > 0
            v_scores = score_matrix[:, f_idx].clone()
            v_scores[~valid_mask] = -9999.0
            
            top_img_indices = torch.topk(v_scores, min(ROWS, valid_mask.sum().item())).indices.cpu().numpy()
            draw.text((col_idx * IMG_SIZE + 5, 10), f"F: {f_idx}\nS: {v_scores.max().item():.1f}", fill=(255,255,255), font=font)
            
            for row_idx, img_idx in enumerate(top_img_indices):
                img_name = sampled_images[img_idx]
                JSON_DATA[method][layer_key][f"feature_{f_idx}"].append({"image_name": img_name, "score": round(v_scores[img_idx].item(), 3)})
                
                try:
                    img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
                    w, h = img.size; s = min(w, h)
                    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((IMG_SIZE, IMG_SIZE))
                    canvas.paste(img, (col_idx * IMG_SIZE, HEADER_H + (row_idx * IMG_SIZE)))
                except: pass
        canvas.save(os.path.join(DIRS[method], f"layer_{layer:02d}_{method}.jpg"), quality=90)

    for L in LAYERS:
        print(f"\nProcessing Layer {L}...")
        sae = AutoEncoderTopK(1024, 65536, 64).to(DEVICE)
        sae.load_state_dict(torch.load(os.path.join(SAE_DIR, f"ae_layer{L}_topk64.pt"), map_location=DEVICE))
        sae.eval()
        
        layer_acts = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sampled_images), BATCH_SIZE)):
                batch_paths = sampled_images[i:i+BATCH_SIZE]
                raw_imgs = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in batch_paths]
                inputs = proc(images=raw_imgs, return_tensors="pt").to(DEVICE)
                h = vision_tower(**inputs, output_hidden_states=True).hidden_states[L]
                acts = sae.encode(h[:, 1:, :].float()).max(dim=1).values
                layer_acts.append(acts.cpu())
                
        all_acts = torch.cat(layer_acts, dim=0)
        freqs = (all_acts > 0).float().sum(dim=0)
    

        stats = {
            "max": all_acts,
            "sum": all_acts, 
            "tfidf": all_acts * torch.log(SAMPLE_SIZE / freqs.clamp(min=1)),
        }
        
        tops = {
            "max": torch.topk(stats["max"].max(dim=0).values, COLS).indices,
            "sum": torch.topk(stats["sum"].sum(dim=0), COLS).indices,
            "tfidf": torch.topk(stats["tfidf"].sum(dim=0), COLS).indices,
        }

        for m in METHODS:
            save_poster(m, L, tops[m], stats[m], all_acts)
            top10_scores = stats[m][:, tops[m]]
            best_local_idx = torch.argmax(top10_scores, dim=1)
            for img_idx in range(SAMPLE_SIZE):
                feat_id = tops[m][best_local_idx[img_idx]].item()
                is_active = all_acts[img_idx, tops[m][best_local_idx[img_idx]]] > 0
                sankey_data[sampled_images[img_idx]][f"L{L}_{m}"] = feat_id if is_active else -1

        del sae; torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(list(sankey_data.values()))
    df.to_csv(CSV_PATH, index=False)
    for m in METHODS:
        with open(JSON_PATHS[m], "w") as f:
            json.dump(JSON_DATA[m], f, indent=4)

fig = go.Figure()

def hex_to_rgba(hex_code, alpha=0.3):
    hex_code = hex_code.lstrip('#')
    if len(hex_code) != 6: return f"rgba(200,200,200,{alpha})"
    r, g, b = int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

for m_idx, method in enumerate(METHODS):
    nodes, node_dict, node_idx = [], {}, 0
    layer_colors = {2:"#1f77b4", 4:"#aec7e8", 6:"#2ca02c", 8:"#98df8a", 10:"#ff7f0e", 12:"#ffbb78", 14:"#9467bd", 16:"#c5b0d5", 18:"#d62728", 20:"#ff9896", 22:"#8c564b"}
    node_color_list = []

    for L in LAYERS:
        for f in poster_features[method][L]:
            name = f"L{L}|F{f}"
            nodes.append(name); node_dict[name] = node_idx
            node_color_list.append(layer_colors.get(L, "grey")); node_idx += 1
        nodes.append(f"L{L}|Other"); node_dict[f"L{L}|Other"] = node_idx
        node_color_list.append("#E0E0E0"); node_idx += 1

    sources, targets, values, link_colors = [], [], [], []
    for i in range(len(LAYERS)-1):
        src_L, tgt_L = LAYERS[i], LAYERS[i+1]
        path_counts = Counter()
        for _, row in df.iterrows():
            s_f, t_f = int(row[f"L{src_L}_{method}"]), int(row[f"L{tgt_L}_{method}"])
            s_n = f"L{src_L}|F{s_f}" if s_f in poster_features[method][src_L] else f"L{src_L}|Other"
            t_n = f"L{tgt_L}|F{t_f}" if t_f in poster_features[method][tgt_L] else f"L{tgt_L}|Other"
            path_counts[(node_dict[s_n], node_dict[t_n])] += 1
            
        for (s, t), count in path_counts.items():
            if count >= 2:
                sources.append(s); targets.append(t); values.append(count)
                link_colors.append(hex_to_rgba(node_color_list[s]))

    fig.add_trace(go.Sankey(
        visible=(m_idx == 0),
        node=dict(pad=15, thickness=20, label=nodes, color=node_color_list),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    ))

buttons = []
for i, m in enumerate(METHODS):
    visible = [False] * len(METHODS)
    visible[i] = True
    buttons.append(dict(label=m.upper(), method="update", args=[{"visible": visible}, {"title": f"Neural Flow Dashboard: {m.upper()}"}]))

fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=0.1, y=1.15)], title_text=f"Neural Flow Dashboard: {METHODS[0].upper()}", width=1800, height=900)
fig.write_html(DASHBOARD_HTML)

print(f"Dashboard Complete: {DASHBOARD_HTML}")