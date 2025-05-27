import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(path):
    img = image.load_img(path, target_size=(224, 224))
    return np.array(img)

def apply_edge_detection(img):
    return cv2.Canny(img, 100, 200)

def apply_sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_sepia(img):
    img = np.array(img, dtype=np.float64)
    sepia = cv2.transform(img, np.matrix([[0.393, 0.769, 0.189],
                                          [0.349, 0.686, 0.168],
                                          [0.272, 0.534, 0.131]]))
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

def apply_deep_fry(img):
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    noise = np.random.randint(0, 50, img.shape, dtype='uint8')
    deepfried = cv2.add(img, noise)
    return deepfried

if __name__ == "__main__":
    original = load_image("dog1.jpg")

    edge = apply_edge_detection(original)
    sharpened = apply_sharpening(original)
    sepia = apply_sepia(original)
    deepfried = apply_deep_fry(original)

    cv2.imwrite("filter_edge.jpg", edge)
    cv2.imwrite("filter_sharpened.jpg", sharpened)
    cv2.imwrite("filter_sepia.jpg", sepia)
    cv2.imwrite("filter_deepfried.jpg", deepfried)

    print("✅ Filters applied and images saved.")
