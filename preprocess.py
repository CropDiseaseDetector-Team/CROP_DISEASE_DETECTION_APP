import cv2
import os

input_dir = "datasets/PlantVillage"  # Adjusted path
output_dir = "datasets/PlantVillage_Processed"

# Supported image extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    
    # Skip if not a directory
    if not os.path.isdir(class_path):
        print(f"⚠️ Skipping non-directory: {class_path}")
        continue
        
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Skip directories and non-image files
        if not os.path.isfile(img_path):
            print(f"⚠️ Skipping non-file: {img_path}")
            continue
            
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            print(f"⚠️ Skipping non-image: {img_path}")
            continue
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Failed to read (corrupt?): {img_path}")
                continue
                
            img = cv2.resize(img, (224, 224))
            output_path = os.path.join(output_dir, class_name, img_name)
            cv2.imwrite(output_path, img)
            print(f"✓ Processed: {img_path}")
            
        except Exception as e:
            print(f"🔥 Error processing {img_path}: {str(e)}")