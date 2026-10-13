import cv2
import numpy as np
from skimage import filters
import glob
def compute_saliency(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    saliency = filters.sobel(gray)  # Simple edge-based saliency
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    return saliency

# Save saliency maps for all images
for img_path in glob.glob("fused_images_human_boosted/**/*.jpg", recursive=True):
    saliency = compute_saliency(img_path)
    cv2.imwrite(img_path.replace(".jpg", "_saliency.png"), saliency.astype('uint8'))
