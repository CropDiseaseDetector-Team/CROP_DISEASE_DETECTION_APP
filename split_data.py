from sklearn.model_selection import train_test_split
import os
import shutil

dataset_path = "datasets/PlantVillage_Processed"
output_base = "datasets/split"

for class_name in os.listdir(dataset_path):
    # Get all image paths for the class
    images = [f"{dataset_path}/{class_name}/{img}" for img in os.listdir(f"{dataset_path}/{class_name}")]
    
    # Split 80/10/10
    train, temp = train_test_split(images, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Save to train/val/test folders
    for split, paths in [("train", train), ("val", val), ("test", test)]:
        os.makedirs(f"{output_base}/{split}/{class_name}", exist_ok=True)
        for img_path in paths:
            shutil.copy(img_path, f"{output_base}/{split}/{class_name}/{os.path.basename(img_path)}")

import tensorflow as tf

# Load pre-trained model
model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Test on a sample image
img = tf.keras.preprocessing.image.load_img("datasets/split/train/Tomato_Bacterial_spot/0a6d40e4-75d6-4659-8bc1-22f47cdb2ca8___GCREC_Bact.Sp 6247.JPG", target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
predictions = model.predict(img_array)
print(predictions.shape)  # Should print (1, 7, 7, 576)