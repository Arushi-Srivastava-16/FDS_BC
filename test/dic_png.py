import pandas as pd

# Input and output CSV
input_csv = "balanced_finding_annotations.csv"
output_csv = "rescaled_boxes_912x1520.csv"

# New dimensions
TARGET_WIDTH = 912
TARGET_HEIGHT = 1520

# Load CSV
df = pd.read_csv(input_csv)

# Normalize first
df['xmin_norm'] = df['xmin'] / df['width']
df['xmax_norm'] = df['xmax'] / df['width']
df['ymin_norm'] = df['ymin'] / df['height']
df['ymax_norm'] = df['ymax'] / df['height']

# Scale to new size
df['xmin_rescaled'] = (df['xmin_norm'] * TARGET_WIDTH).round(2)
df['xmax_rescaled'] = (df['xmax_norm'] * TARGET_WIDTH).round(2)
df['ymin_rescaled'] = (df['ymin_norm'] * TARGET_HEIGHT).round(2)
df['ymax_rescaled'] = (df['ymax_norm'] * TARGET_HEIGHT).round(2)

# Save to CSV
df.to_csv(output_csv, index=False)

print(f"Rescaled bounding boxes saved to {output_csv}")