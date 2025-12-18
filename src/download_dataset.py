# Example script to download and prepare PlantVillage dataset
# This is a template - you'll need to download the dataset from Kaggle first

"""
Download PlantVillage Dataset from Kaggle:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

After downloading, extract to data/raw/
"""

import os
import shutil
from pathlib import Path

def organize_dataset(source_dir, target_dir):
    """
    Organize downloaded dataset into train/val structure
    
    Args:
        source_dir: Path to downloaded dataset
        target_dir: Path to organized dataset (data/raw)
    """
    print(f"Organizing dataset from {source_dir} to {target_dir}")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy class directories
    for class_dir in Path(source_dir).iterdir():
        if class_dir.is_dir():
            target_class_dir = os.path.join(target_dir, class_dir.name)
            if not os.path.exists(target_class_dir):
                shutil.copytree(class_dir, target_class_dir)
                print(f"Copied {class_dir.name}")
    
    print("Dataset organization complete!")


if __name__ == "__main__":
    # Example usage
    source = "path/to/downloaded/PlantVillage"  # Update this path
    target = "data/raw"
    
    if os.path.exists(source):
        organize_dataset(source, target)
    else:
        print(f"Source directory not found: {source}")
        print("Please download PlantVillage dataset from Kaggle")
