import pandas as pd
import plotly.express as px
import os

# Create output directory for deliverables
os.makedirs('output', exist_ok=True)

# Load the data
df = pd.read_csv(r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\Micro_changed_50_withoutPCA\clustering_scores_comparison.csv")

# Ensure Clusters is treated as a categorical variable to enable grouped discrete plotting
df['Clusters'] = df['Clusters'].astype(str)
cluster_order = ["5", "10", "50", "100", "500"]

def generate_grouped_plot(dataframe, metric_name, output_dir='output'):
    """
    Plots the given metric for each layer, grouped by the number of clusters.
    This creates the structure: Layer1(K5, K10...), [space], Layer2(K5, K10...)
    """
    # Create a grouped bar chart
    fig = px.bar(
        dataframe,
        x='Layer',
        y=metric_name,
        color='Clusters',
        barmode='group',
        title=f'{metric_name} across Layers grouped by Number of Clusters (K)',
        category_orders={"Clusters": cluster_order},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    # Clean up the layout
    fig.update_layout(
        xaxis=dict(
            tickmode='linear', 
            dtick=1, # Ensure every layer is labeled
            title='Neural Network Layer'
        ),
        yaxis_title=metric_name,
        legend_title="Clusters (K)",
        plot_bgcolor='rgba(245, 245, 245, 1)',  # Light gray background for contrast
        margin=dict(l=50, r=20, t=60, b=50)
    )
    
    # Save the figure
    safe_metric_name = metric_name.lower().replace("-", "_")
    output_path = f'{output_dir}/{safe_metric_name}_by_layer.png'
    fig.write_image(output_path, width=1400, height=600, scale=2)
    print(f"Saved: {output_path}")
    
    return fig

# Generate and save a plot for each metric
metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Dunn']

for metric in metrics:
    generate_grouped_plot(df, metric)
