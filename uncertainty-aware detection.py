import os
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import DetrForObjectDetection, DetrImageProcessor

# --- Paths ---
fused_image_dir = 'Fused_Images_human_boosted'
output_dir = 'result_final'
ckpt_dir = 'detr_finetuned_ckpt'
os.makedirs(output_dir, exist_ok=True)

# --- Load fine-tuned model and processor ---
model = DetrForObjectDetection.from_pretrained(ckpt_dir)
processor = DetrImageProcessor.from_pretrained(ckpt_dir)
model.eval()

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

MC_SAMPLES = 10
DETECTION_THRESHOLD = 0.7
LABEL_MAP = {1: "human", 2: "car"}
COLOR_MAP = {1: (0, 255, 0), 2: (255, 0, 0)}

def detect_with_uncertainty(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    all_boxes, all_scores, all_labels = [], [], []

    for _ in range(MC_SAMPLES):
        enable_mc_dropout(model)
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=[image.size[::-1]],
            threshold=DETECTION_THRESHOLD
        )[0]
        all_boxes.append(results["boxes"].cpu().numpy())
        all_scores.append(results["scores"].cpu().numpy())
        all_labels.append(results["labels"].cpu().numpy())

    # Stack and compute statistics
    all_boxes = np.stack(all_boxes)
    all_scores = np.stack(all_scores)
    all_labels = np.stack(all_labels)

    mean_boxes = np.mean(all_boxes, axis=0)
    std_boxes = np.std(all_boxes, axis=0)
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    mean_labels = np.round(np.mean(all_labels, axis=0)).astype(int)

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in range(mean_boxes.shape[0]):
        x1, y1, x2, y2 = map(int, mean_boxes[i])
        label = mean_labels[i]
        score = mean_scores[i]
        box_std = np.mean(std_boxes[i])
        score_std = std_scores[i]

        if label not in LABEL_MAP:
            continue

        color = COLOR_MAP[label]
        label_name = LABEL_MAP[label]
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_cv, 
            f"{label_name} {score:.2f}±{score_std:.2f}", 
            (x1, max(0, y1-10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, color, 2
        )
        cv2.putText(
            image_cv, 
            f"loc std: {box_std:.1f}", 
            (x1, min(image_cv.shape[0]-10, y2+20)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0,0,255), 1
        )
    return image_cv

def main():
    # Test on first 5 images
    test_files = [f for f in os.listdir(fused_image_dir) if f.lower().endswith(".jpg")][:15]
    for fname in test_files:
        image_path = os.path.join(fused_image_dir, fname)
        result_img = detect_with_uncertainty(image_path)
        out_path = os.path.join(output_dir, f"TEST_{fname}")
        cv2.imwrite(out_path, result_img)
        print(f"Tested {fname}")


if __name__ == "__main__":
    main()
    print("Uncertainty-aware detection complete.")
