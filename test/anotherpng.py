import pandas as pd

# Input and output CSV
input_csv = "rescaled_boxes_912x1520.csv"
output_csv = "final_with_image_paths.csv"

# Load CSV
df = pd.read_csv(input_csv)

# Create image path column
df.insert(0, 'image_path', df['study_id'] + "/" + df['image_id'] + ".png")

# Save updated CSV
df.to_csv(output_csv, index=False)

print(f"Final CSV with image_path saved to {output_csv}")