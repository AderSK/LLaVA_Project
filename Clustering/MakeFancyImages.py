from PIL import Image
import os

# 1. Define your specific image paths in order
# Make sure the heatmap at index 0 corresponds to the original at index 0
original_paths = [r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000068381.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000133680.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000027519.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000561780.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000430774.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000435444.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000527220.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000259342.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000263664.jpg",
r"C:\Users\samko\Desktop\Bakalarka\ManipulatedDataset\COCO_val2014_000000390348.jpg",
]

heatmap_paths = [r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000068381.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000133680.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000027519.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000561780.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000430774.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000435444.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000527220.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000259342.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000263664.jpg",
r"C:\Users\samko\Desktop\Bakalarka\LLaVA_Project\Clustering\heatmaps\heatmap_L10_C10_COCO_val2014_000000390348.jpg",

]

# 2. Load the images
row1_images = [Image.open(path) for path in original_paths]
row2_images = [Image.open(path) for path in heatmap_paths]

# Combine them: The first 5 go to the top row, the next 5 go to the bottom row
all_images = row1_images + row2_images

# 3. Define grid dimensions
cols = 10
rows = 2

# 4. Resize everything to match (using the first original image's size as the standard)
target_width, target_height = all_images[0].size
all_images = [img.resize((target_width, target_height)) for img in all_images]

# 5. Create a blank canvas for the final grid
grid_width = cols * target_width
grid_height = rows * target_height
grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

# 6. Paste each image into the correct position
for index, img in enumerate(all_images):
    col = index % cols
    row = index // cols
    
    x = col * target_width
    y = row * target_height
    
    grid_image.paste(img, (x, y))

# 7. Save the final result
output_path = "concept_heatmap_comparison.png"
grid_image.save(output_path)
print(f"Saved ordered grid to {output_path}")