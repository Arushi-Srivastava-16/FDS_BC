import pandas as pd

# Load the CSV file
df = pd.read_csv("finding_annotations.csv")

# Define what counts as a positive finding
def has_finding(row):
    val = str(row['finding_categories']).strip()
    return val != "['No Finding']" and val != "[]" and val != "" and val.lower() != "nan"

# Mark whether each row has a real finding
df['has_finding'] = df.apply(has_finding, axis=1)

# Get all positive study_ids (at least one row has a finding)
positive_study_ids = df[df['has_finding']]['study_id'].unique()

# Get all rows for positive study_ids
positive_df = df[df['study_id'].isin(positive_study_ids)]

# Get all study_ids in the dataset
all_study_ids = df['study_id'].unique()

# Determine negative study_ids (never have any findings)
negative_study_ids = list(set(all_study_ids) - set(positive_study_ids))
negative_df = df[df['study_id'].isin(negative_study_ids)]

# Check if there are enough negatives to sample from
if len(negative_study_ids) < len(positive_study_ids):
    raise ValueError("Not enough negative samples to balance positives.")

# Sample an equal number of negative study_ids (1:1 ratio)
sampled_negative_study_ids = pd.Series(negative_study_ids).sample(
    n=len(positive_study_ids), random_state=42
)

# Get all rows for the sampled negative study_ids
sampled_negative_df = negative_df[negative_df['study_id'].isin(sampled_negative_study_ids)]

# Combine positive and sampled negative datasets
balanced_df = pd.concat([positive_df, sampled_negative_df])

# Shuffle the combined dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV
balanced_df.to_csv("balanced_finding_annotations.csv", index=False)

print(f"Balanced dataset created with {len(positive_study_ids)} positive and {len(sampled_negative_study_ids)} negative studies.")
print(f"Total rows in final dataset: {len(balanced_df)}")