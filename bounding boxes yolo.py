import roboflow

# Replace with your Roboflow API key
rf = roboflow.Roboflow(api_key="api key")
workspace = rf.workspace("Research Project")
project = workspace.project("Yolo Visible-Infrared")

# Upload images from a folder
image_folder = "C:/Users/shiva/Research Project/YOLO_dataset/train/images"
for img_file in os.listdir(image_folder):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        project.upload(os.path.join(image_folder, img_file))
