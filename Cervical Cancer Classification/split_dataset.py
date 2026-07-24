import os
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

# Paths
source_dir = "archive"
destination_dir = "data/SIPaKMeD"

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Loop through each class
for class_name in os.listdir(source_dir):

    class_path = os.path.join(
        source_dir,
        class_name,
        class_name,
        "CROPPED"
    )

    # Skip if the folder doesn't exist
    if not os.path.isdir(class_path):
        continue

    # Get only .bmp image files
    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(".bmp")
    ]

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(
            os.path.join(destination_dir, split, class_name),
            exist_ok=True
        )

    # Copy training images
    for image in train_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(destination_dir, "train", class_name, image)
        )

    # Copy validation images
    for image in val_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(destination_dir, "val", class_name, image)
        )

    # Copy test images
    for image in test_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(destination_dir, "test", class_name, image)
        )

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} val, "
        f"{len(test_images)} test "
        f"(Total: {total})"
    )

print("\nDataset successfully split!")