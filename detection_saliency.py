import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm

# --- Custom Dataset: Fused + Saliency as 4-channel ---
class FusedSaliencyCocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, sal_folder, transform=None):
        super().__init__(img_folder, ann_file)
        self.sal_folder = sal_folder
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        sal_path = os.path.join(self.sal_folder, img_name)
        sal_map = Image.open(sal_path).convert('L').resize(img.size)
        img_np = np.array(img)
        sal_np = np.array(sal_map)[..., None]
        img4 = np.concatenate([img_np, sal_np], axis=-1)  # H x W x 4
        if self.transform:
            img4 = self.transform(img4)
        else:
            img4 = torch.from_numpy(img4).permute(2, 0, 1).float() / 255.0
        return img4, target, img_name, img.size  # Return img_name and size for saving

def coco_collate_fn(batch):
    return list(zip(*batch))

def modify_detr_input_channels(model, in_channels=4):
    old_conv = model.model.backbone.conv.encoder.conv
    new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=old_conv.bias is not None)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        nn.init.kaiming_normal_(new_conv.weight[:, 3:].unsqueeze(1))
        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias
    model.model.backbone.conv1 = new_conv
    return model

def save_images_with_boxes(model, processor, dataset, device, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model.eval()
    font = ImageFont.load_default()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Saving labeled images"):
            img4, _, img_name, img_size = dataset[i]
            pixel_values = img4.unsqueeze(0).to(device)
            outputs = model(pixel_values=pixel_values)
            results = processor.post_process_object_detection(
                outputs, threshold=0.7, target_sizes=[img_size]
            )[0]
            # Load original RGB image for visualization
            img_path = os.path.join(dataset.root, img_name)
            orig_img = Image.open(img_path).convert('RGB')
            draw = ImageDraw.Draw(orig_img)
            for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
                box = [int(x) for x in box]
                label_name = model.config.id2label[label.item()]
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]), f"{label_name} {score:.2f}", fill="red", font=font)
            orig_img.save(os.path.join(output_folder, img_name))
    model.train()

def main():
    img_folder = 'Fused_Images_human_boosted'
    ann_file = 'coco_annotations - Copy.json'
    sal_folder = 'saliency_maps'
    output_dir = 'detr_fewshot_saliency_ckpt'
    labeled_img_dir = 'label_saliency_images'
    os.makedirs(output_dir, exist_ok=True)

    dataset = FusedSaliencyCocoDataset(img_folder, ann_file, sal_folder)
    few_shot_ids = list(range(len(dataset)))
    train_dataset = torch.utils.data.Subset(dataset, few_shot_ids)
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=coco_collate_fn)

    id2label = {1: "human", 2: "car"}
    label2id = {"human": 1, "car": 2}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model = modify_detr_input_channels(model, in_channels=4).to(device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for images, targets, _, _ in tqdm(loader):
            pixel_values = torch.stack(images).to(device)
            labels = []
            for target in targets:
                if not target:
                    continue
                target_dict = {
                    'image_id': target[0]['image_id'],
                    'annotations': target
                }
                encoding = processor(images=None, annotations=target_dict, return_tensors="pt")
                labels.append(encoding["labels"][0])
            if not labels:
                continue
            labels = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in l.items()} for l in labels]
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item():.4f}")

        # Save checkpoint
        model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))
        processor.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))

        # Save bounding box images after each epoch
        save_images_with_boxes(model, processor, dataset, device, labeled_img_dir)

    print("Training complete.")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
