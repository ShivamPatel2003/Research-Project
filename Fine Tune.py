import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm

def coco_collate_fn(batch):
    return list(zip(*batch))

def main():
    # --- Paths ---
    img_dir = 'training_labels/label_images'
    ann_file = 'coco_annotations.json'
    output_dir = 'detr_finetuned_ckpt'
    os.makedirs(output_dir, exist_ok=True)

    # --- Hyperparameters ---
    batch_size = 4
    num_workers = 4
    num_epochs = 10
    learning_rate = 5e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load dataset and DataLoader ---
    dataset = CocoDetection(img_dir, ann_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=coco_collate_fn
    )

    # --- Load DETR and processor ---
    id2label = {1: "human", 2: "car"}
    label2id = {"human": 1, "car": 2}
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Training loop ---
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in tqdm(loader):
            pixel_values_list = []
            labels_list = []
            for img, target in zip(images, targets):
                if not target:  # Skip images with no annotations
                    continue
                target_dict = {'image_id': target[0]['image_id'], 'annotations': target}
                encoding = processor(images=img, annotations=target_dict, return_tensors="pt")
                pixel_values_list.append(encoding["pixel_values"].squeeze(0))
                labels_list.append(encoding["labels"][0])
            if not pixel_values_list:
                continue  # Skip batch if all images had no annotations
            pixel_values = torch.stack(pixel_values_list).to(device)
            labels = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in l.items()} for l in labels_list]
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item():.4f}")

        # Save checkpoint
        model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))
        processor.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))

    print("Fine-tuning complete.")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
