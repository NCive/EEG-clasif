import pandas as pd
import os

input_folder = "processed_subjects"
output_folder = "output_parquet_files"

os.makedirs(output_folder, exist_ok=True)

block_size = 125

# Define label columns
label_cols = ["Valence", "Arousal", "Dominance", "Liking"]

for root, dirs, files in os.walk(input_folder):
    print("Scanning:", root)

    for file in files:
        if file.lower().endswith(".csv"):
            
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Keep numeric columns
            df = df.select_dtypes(include='number')
            
            # Split features and labels
            feature_df = df.drop(columns=label_cols, errors='ignore')
            label_df = df[label_cols]
            
            all_blocks = []

            # ✅ NON-overlapping windows
            for start in range(0, len(df), block_size):
                block_features = feature_df.iloc[start:start + block_size]
                block_labels = label_df.iloc[start:start + block_size]
                
                # Skip incomplete blocks
                if len(block_features) < block_size:
                    continue
                
                stats_dict = {}

                # Compute stats ONLY on features
                for col in block_features.columns:
                    stats_dict[f"{col}_mean"] = block_features[col].mean()
                    stats_dict[f"{col}_std"] = block_features[col].std()
                    stats_dict[f"{col}_min"] = block_features[col].min()
                    stats_dict[f"{col}_max"] = block_features[col].max()

                # ✅ Add labels (aggregated per block)
                stats_dict["Valence"] = block_labels["Valence"].mean()
                stats_dict["Arousal"] = block_labels["Arousal"].mean()
                stats_dict["Dominance"] = block_labels["Dominance"].mean()
                stats_dict["Liking"] = block_labels["Liking"].mean()

                all_blocks.append(stats_dict)

            if not all_blocks:
                continue

            stats_df = pd.DataFrame(all_blocks)
            
            folder_name = os.path.basename(root)
            output_name = f"{folder_name}_{file.replace('.csv', '')}.parquet"
            parquet_file = os.path.join(output_folder, output_name)
            
            stats_df.to_parquet(parquet_file, index=False)

            print(f"{file} → {len(stats_df)} blocks → saved")