import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Paths (update these for your system)
CSV_PATH = "rescaled_boxes_912x1520.csv"
IMAGE_ROOT = "vindrpng/images_png"  # Path to folder containing subfolders (study_id/image_id.png)
OUTPUT_CROP_DIR = "cropped_images_simclr"
OUTPUT_CSV = "crop_metadata.csv"

# Crop size for negative samples
CROP_SIZE = 224

# Image transformation to resize crop
resize_crop = transforms.Compose([
    transforms.Resize((224, 224))
])

# Load CSV
df = pd.read_csv(CSV_PATH)

# Save metadata for saved crops
crop_data = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(IMAGE_ROOT, row['study_id'], f"{row['image_id']}.png")
    if not os.path.exists(image_path):
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except:
        continue

    w, h = image.size

    if row['has_finding']:
        # Convert normalized bbox to pixel coordinates
        xmin = float(row['xmin'])
        ymin = float(row['ymin'])
        xmax = float(row['xmax'])
        ymax = float(row['ymax'])
        
        # Crop & resize
        crop = image.crop((xmin, ymin, xmax, ymax))
        crop = resize_crop(crop)

        label = "positive"
    else:
        # Take center crop or top-left negative crop of size 224x224
        crop = image.crop((0, 0, min(w, CROP_SIZE), min(h, CROP_SIZE)))
        crop = resize_crop(crop)
        label = "negative"

    # Save crop
    out_path = os.path.join(OUTPUT_CROP_DIR, label)
    os.makedirs(out_path, exist_ok=True)
    crop_filename = f"{row['study_id']}_{row['image_id']}_{label}_{idx}.png"
    crop.save(os.path.join(out_path, crop_filename))

    crop_data.append({
        "filename": crop_filename,
        "label": label,
        "view": row['view_position'],
        "laterality": row['laterality'],
        "study_id": row['study_id'],
        "image_id": row['image_id']
    })

# Save metadata
pd.DataFrame(crop_data).to_csv(OUTPUT_CSV, index=False)

"Done saving crops and metadata."