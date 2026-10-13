import os
import numpy as np
import gc
from glob import glob
from multiprocessing import Pool, cpu_count, freeze_support

# Define Paths
train_rgb_path = r"C:\Users\shiva\OneDrive\Desktop\FLIR_ADAS_1_3\train\RGB"
train_thermal_path = r"C:\Users\shiva\OneDrive\Desktop\FLIR_ADAS_1_3\train\thermal_8_bit"
fused_images_path = r"C:\Users\shiva\OneDrive\Desktop\FLIR_ADAS_1_3\train\Fused_Images_human_boosted"

# Get image paths
train_rgb_images = sorted(glob(os.path.join(train_rgb_path, "*.*")))[4613:7613]
train_thermal_images = sorted(glob(os.path.join(train_thermal_path, "*.*")))[4613:7613]
os.makedirs(fused_images_path, exist_ok=True)

def align_images(rgb, thermal):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(rgb, None)
    kp2, des2 = orb.detectAndCompute(thermal, None)
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.7*n.distance]
    if len(good) < 10:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def human_boosted_fusion(rgb, thermal, thermal_weight=0.8, rgb_weight=0.2):
    # Convert to grayscale
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    thermal_gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)

    # Normalize images
    rgb_norm = rgb_gray.astype(np.float32) / 255.0
    thermal_norm = thermal_gray.astype(np.float32) / 255.0

    # Enhance edges in thermal (humans are often strong edges in thermal)
    sobelx = cv2.Sobel(thermal_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(thermal_norm, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = cv2.magnitude(sobelx, sobely)
    edge_map = np.clip(edge_map * 2.0, 0, 1)

    # Boost thermal in likely human temperature range (mid-high gray values)
    thermal_boost = np.where(thermal_gray > 80, 1.5, 1.0)  # tune threshold as needed
    thermal_human = np.clip(thermal_norm * thermal_boost, 0, 1)

    # Weighted fusion
    fused = (thermal_weight * thermal_human + rgb_weight * rgb_norm + 0.2 * edge_map) / (thermal_weight + rgb_weight + 0.2)
    fused = np.clip(fused, 0, 1)

    # Enhance local contrast
    fused_uint8 = (fused * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    fused_clahe = clahe.apply(fused_uint8)

    # Convert to 3-channel for saving
    fused_bgr = cv2.cvtColor(fused_clahe, cv2.COLOR_GRAY2BGR)
    return fused_bgr

def process_single_image(index):
    try:
        rgb = cv2.imread(train_rgb_images[index])
        thermal = cv2.imread(train_thermal_images[index])
        if rgb is None or thermal is None:
            print(f"⚠️ Skipped {index}: Could not load image.")
            return

        # Resize to same dimensions
        h, w = min(rgb.shape[0], thermal.shape[0]), min(rgb.shape[1], thermal.shape[1])
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        thermal = cv2.resize(thermal, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Try alignment
        H = align_images(rgb, thermal)
        if H is not None:
            aligned_rgb = cv2.warpPerspective(rgb, H, (w, h),
                                              flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                                              borderMode=cv2.BORDER_REFLECT)
            alignment_used = True
        else:
            aligned_rgb = rgb
            alignment_used = False

        # Human-boosted fusion
        fused = human_boosted_fusion(aligned_rgb, thermal)
        out_path = os.path.join(fused_images_path, f"fused_{index}.jpg")
        cv2.imwrite(out_path, fused, [cv2.IMWRITE_JPEG_QUALITY, 100])

        print(f"✅ Processed {index+1}/{len(train_rgb_images)} | Alignment: {'Yes' if alignment_used else 'No'} | Saved: {out_path}")

    except Exception as e:
        print(f"⚠️ Error {index}: {str(e)}")
    finally:
        gc.collect()

def process_all_images():
    num_workers = min(cpu_count(), 8)
    print(f"\n🚀 Processing with {num_workers} workers\n")
    with Pool(num_workers) as pool:
        pool.map(process_single_image, range(len(train_rgb_images)))
    print("\n✅ Fusion complete!")

if __name__ == "__main__":
    freeze_support()
    process_all_images()