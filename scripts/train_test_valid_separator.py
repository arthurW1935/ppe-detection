import os
import argparse
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    # Ensure the ratios sum to 1
    assert abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'test', 'valid']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(input_dir, 'images')) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Split the dataset
    train_valid, test = train_test_split(image_files, test_size=test_ratio, random_state=42)
    train, valid = train_test_split(train_valid, test_size=valid_ratio/(train_ratio+valid_ratio), random_state=42)

    # Function to copy files
    def copy_files(files, split):
        for f in files:
            image_src = os.path.join(input_dir, 'images', f)
            label_src = os.path.join(input_dir, 'labels', f.rsplit('.', 1)[0] + '.txt')
            
            image_dst = os.path.join(output_dir, split, 'images', f)
            label_dst = os.path.join(output_dir, split, 'labels', f.rsplit('.', 1)[0] + '.txt')
            
            shutil.copy(image_src, image_dst)
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)

    # Copy files to respective directories
    copy_files(train, 'train')
    copy_files(test, 'test')
    copy_files(valid, 'valid')

    # Print statistics
    print(f"Total images: {len(image_files)}")
    print(f"Training set: {len(train)} images")
    print(f"Testing set: {len(test)} images")
    print(f"Validation set: {len(valid)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train, test, and validation sets')
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing images and labels subdirectories')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for the split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training set (default: 0.7)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of test set (default: 0.2)')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Ratio of validation set (default: 0.1)')
    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir, args.train_ratio, args.test_ratio, args.valid_ratio)
    print("Dataset splitting completed.")