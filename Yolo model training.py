import cv2
import os

# --- Configuration ---
IMAGE_DIR = r"C:\Users\shiva\Research Project\Fused_Images_human_boosted"
LABEL_DIR = r"C:\Users\shiva\Research Project\yolo_dataset\train\labels"
CLASS_MAP = {'h': 0, 'c': 1}  # YOLO classes: 0 = human, 1 = car

os.makedirs(LABEL_DIR, exist_ok=True)

all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")]
all_images.sort()  # Optional: process in order

drawing = False
ix, iy = -1, -1
boxes = []
current_label = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_label, boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if current_label is not None:
            boxes.append((current_label, x1, y1, x2, y2))
            print(f"Added box: {current_label}, {x1}, {y1}, {x2}, {y2}")

def label_images():
    global current_label, boxes
    for fname in all_images:
        image_path = os.path.join(IMAGE_DIR, fname)
        image = cv2.imread(image_path)
        if image is None:
            continue

        boxes = []
        print(f"\n--- Annotating {fname} ---")
        print("Press 'h' for human, 'c' for car BEFORE drawing each box.")
        print("Draw boxes with mouse. Press 's' to save & next image, 'r' to reset boxes, 'q' to quit.")

        clone = image.copy()
        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", draw_rectangle)

        while True:
            temp_img = clone.copy()
            for label, x1, y1, x2, y2 in boxes:
                color = (0, 255, 0) if label == 0 else (255, 0, 0)
                cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
            cv2.imshow("Annotator", temp_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                current_label = 0
                print("Label set to: Human")
            elif key == ord("c"):
                current_label = 1
                print("Label set to: Car")
            elif key == ord("r"):
                boxes = []
                print("Boxes reset.")
            elif key == ord("s"):
                # Save YOLO label
                label_path = os.path.join(LABEL_DIR, fname.replace(".jpg", ".txt"))
                h, w = image.shape[:2]
                with open(label_path, "w") as f:
                    for label, x1, y1, x2, y2 in boxes:
                        # Convert to YOLO format: x_center, y_center, width, height (normalized)
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                print(f"Saved: {label_path}")
                break
            elif key == ord("q"):
                print("Quitting annotation.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

if __name__ == "__main__":
    label_images()
