import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

model = MobileNetV2(weights='imagenet')

def classify_and_show(image_array, label):
    img_array = preprocess_input(image_array.copy())
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    top_preds = decode_predictions(predictions, top=3)[0]
    print(f"Top-3 Predictions for {label}:")
    for i, (imagenet_id, lbl, score) in enumerate(top_preds):
        print(f" {i + 1}: {lbl} ({score:.2f})")

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    return image.img_to_array(img)

def apply_black_box(img_array):
    occluded = img_array.copy()
    occluded[70:150, 90:170] = 0  # black box over face
    return occluded

def apply_blur(img_array):
    occluded = img_array.copy()
    face_region = occluded[70:150, 90:170]
    blurred = cv2.GaussianBlur(face_region, (15, 15), 0)
    occluded[70:150, 90:170] = blurred
    return occluded

def apply_pixelation(img_array):
    occluded = img_array.copy()
    face_region = occluded[70:150, 90:170]
    small = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (80, 80), interpolation=cv2.INTER_NEAREST)
    occluded[70:150, 90:170] = pixelated
    return occluded

if __name__ == "__main__":
    img_path = "dog1.jpg"
    original = load_image(img_path)

    black_box_img = apply_black_box(original)
    blur_img = apply_blur(original)
    pixel_img = apply_pixelation(original)

    cv2.imwrite("occluded_black_box.jpg", black_box_img)
    cv2.imwrite("occluded_blur.jpg", blur_img)
    cv2.imwrite("occluded_pixel.jpg", pixel_img)

    print("✅ Images saved. Now classifying each...\n")

    classify_and_show(black_box_img, "Black Box")
    classify_and_show(blur_img, "Blurred Face")
    classify_and_show(pixel_img, "Pixelated Face")
