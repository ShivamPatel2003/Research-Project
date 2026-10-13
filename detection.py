import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

# --- Configuration ---
IMAGE_DIR = r"C:\Users\shiva\OneDrive\Desktop\FLIR_ADAS_1_3\train\Fused_Images_human_boosted"
OUTPUT_DIR = "model_labels"
ANNOTATION_DIR = os.path.join(OUTPUT_DIR, "label_annotations")
IMAGE_SAVE_DIR = os.path.join(OUTPUT_DIR, "label_images")
os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load pre-trained DETR model (ViT-based)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Class mapping (COCO dataset classes - person=1, car=3)
CLASS_MAPPING = {
    1: ("human", (0, 255, 0)),   # Green for humans
    3: ("car", (255, 0, 0))      # Red for cars
}

def detect_and_save():
    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
            
        image_path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(image_path).convert("RGB")
        
        # Process image through model
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process detections (threshold=0.7)
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=[image.size[::-1]],
            threshold=0.7
        )[0]

        # Convert to OpenCV format for drawing
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        annotations = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() not in CLASS_MAPPING:
                continue
                
            # Get detection info
            label_name, color = CLASS_MAPPING[label.item()]
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # Draw bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, 
                       f"{label_name} {score:.2f}", 
                       (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
            
            # Save annotation
            annotations.append({
                "bbox": [x1, y1, x2-x1, y2-y1],
                "label": label.item(),
                "confidence": float(score)
            })

        # Save labeled image
        output_image_path = os.path.join(IMAGE_SAVE_DIR, fname)
        cv2.imwrite(output_image_path, image_np)
        
        # Save annotations
        annotation_path = os.path.join(ANNOTATION_DIR, fname.replace(".jpg", ".json"))
        with open(annotation_path, "w") as f:
            json.dump({
                "file_name": fname,
                "annotations": annotations
            }, f, indent=2)
            
        print(f"Processed {fname} - Found {len(annotations)} objects")

if __name__ == "__main__":
    detect_and_save()
    print(f"\n✅ Auto-labeling complete! Check {OUTPUT_DIR} for results")
