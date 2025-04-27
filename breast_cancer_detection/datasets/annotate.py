import pandas as pd
import os

def create_vindr_annotations(image_root_dir, finding_csv_path, output_csv_path):
    """
    Create a clean annotations CSV mapping images to bounding boxes.
    
    Args:
        image_root_dir (str): Path to the folder containing study_id/image_id.png
        finding_csv_path (str): Path to the finding.csv provided by VinDr.
        output_csv_path (str): Where to save the generated annotations CSV.
    """

    # Load the finding.csv
    findings = pd.read_csv(finding_csv_path)

    # We'll store the final annotations here
    annotations = []

    for idx, row in findings.iterrows():
        study_id = row['study_id']
        image_id = row['image_id']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']

        # Build relative path: study_id/image_id.png
        img_path = os.path.join(study_id, f"{image_id}.png")

        # Check if bbox is valid
        if not any(pd.isna([xmin, ymin, xmax, ymax])):
            annotations.append({
                'image_path': img_path,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
    
    # Create DataFrame
    annotations_df = pd.DataFrame(annotations)

    # Save it
    annotations_df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(annotations_df)} annotations to {output_csv_path}")

