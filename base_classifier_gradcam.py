import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the model with pretrained weights
model = MobileNetV2(weights='imagenet')
grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer("Conv_1").output, model.output]
)

def compute_gradcam(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Forward pass and gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Gradients of the top predicted class wrt conv layer
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each conv filter by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), img

def overlay_heatmap(heatmap, original_img, output_path="heatmap_dog1.jpg", alpha=0.4):
    img = np.array(original_img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed_img = heatmap_color * alpha + img
    cv2.imwrite(output_path, overlayed_img)

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    top_preds = decode_predictions(preds, top=3)[0]

    print("Top-3 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(top_preds):
        print(f"{i + 1}: {label} ({score:.2f})")

if __name__ == "__main__":
    image_path = "dog1.jpg"
    classify_image(image_path)
    heatmap, original_img = compute_gradcam(image_path)
    overlay_heatmap(heatmap, original_img)
    print("✅ Grad-CAM heatmap saved as heatmap_dog1.jpg")
