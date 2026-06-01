import pandas as pd
import plotly.graph_objects as go

CSV_PATH = "feature_paths_all_layers.csv"
OUTPUT_HTML = "concept_flow_11_layers.html"
LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

MIN_PATH_COUNT = 1

print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

nodes = []
node_dict = {}
node_idx = 0
node_colors = []

layer_colors = {
    2: "#1f77b4",  4: "#aec7e8",
    6: "#2ca02c",  8: "#98df8a",
    10: "#ff7f0e", 12: "#ffbb78",
    14: "#9467bd", 16: "#c5b0d5",
    18: "#d62728", 20: "#ff9896",
    22: "#8c564b"
}

for L in LAYERS:
    col_name = f"L{L}_feature"
    unique_features = df[col_name].unique()
    for feat in unique_features:
        node_name = f"L{L} | F{int(feat)}"
        nodes.append(node_name)
        node_dict[node_name] = node_idx
        node_colors.append(layer_colors.get(L, "grey"))
        node_idx += 1

sources, targets, values = [], [], []

for i in range(len(LAYERS) - 1):
    source_layer = LAYERS[i]
    target_layer = LAYERS[i+1]
    
    source_col = f"L{source_layer}_feature"
    target_col = f"L{target_layer}_feature"
    
    transitions = df.groupby([source_col, target_col]).size().reset_index(name='count')
    transitions = transitions[transitions['count'] >= MIN_PATH_COUNT]
    
    for _, row in transitions.iterrows():
        src_name = f"L{source_layer} | F{int(row[source_col])}"
        tgt_name = f"L{target_layer} | F{int(row[target_col])}"
        
        sources.append(node_dict[src_name])
        targets.append(node_dict[tgt_name])
        values.append(row['count'])


fig = go.Figure(data=[go.Sankey(
    node=dict(
      pad=20,
      thickness=20,
      line=dict(color="black", width=0.5),
      label=nodes,
      color=node_colors
    ),
    link=dict(
      source=sources,
      target=targets,
      value=values,
      color="rgba(150, 150, 150, 0.4)" 
    )
)])

fig.update_layout(
    title_text="Neural Concept Flow: 11-Layer Lineage Map (TopK=64)",
    font_size=11,
    width=1800, 
    height=900,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.write_html(OUTPUT_HTML)