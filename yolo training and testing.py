import os
import random
import shutil

# Configuration
source_dir = r"C:/Users/shiva/Research Project/Fused_Images_human_boosted"
dest_dir = r"C:/Users/shiva/Research Project/YOLO_dataset"
train_size = 500  # Number of images to use for training/validation
test_size = 2500  # Remaining images for testing
val_split = 0.2  # 20% of training data for validation

# Create directory structure
os.makedirs(os.path.join(dest_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "val", "labels"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "test", "images"), exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(source_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Randomly select images
random.seed(42)  # For reproducibility
train_val_images = random.sample(all_images, train_size)
test_images = list(set(all_images) - set(train_val_images))

# Split train/validation
val_size = int(len(train_val_images) * val_split)
val_images = random.sample(train_val_images, val_size)
train_images = list(set(train_val_images) - set(val_images))

def copy_images(image_list, split_name):
    for img_file in image_list:
        # Copy image
        src = os.path.join(source_dir, img_file)
        dest = os.path.join(dest_dir, split_name, "images", img_file)
        shutil.copy2(src, dest)
        
        # Create empty label file (you'll fill these later)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        open(os.path.join(dest_dir, split_name, "labels", label_file), 'w').close()

# Copy files to respective directories
copy_images(train_images, "train")
copy_images(val_images, "val")

# Copy test images
for img_file in test_images:
    src = os.path.join(source_dir, img_file)
    dest = os.path.join(dest_dir, "test", "images", img_file)
    shutil.copy2(src, dest)

# Create data.yaml
yaml_content = f"""train: {os.path.join(dest_dir, 'train', 'images')}
val: {os.path.join(dest_dir, 'val', 'images')}
test: {os.path.join(dest_dir, 'test', 'images')}

nc: 2
names: ['human', 'car']
"""

with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

print(f"""Dataset prepared:
- Training images: {len(train_images)}
- Validation images: {len(val_images)}
- Test images: {len(test_images)}
""")
