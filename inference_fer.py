import tensorflow as tf
import numpy as np
import cv2

# ---------------------------------------------
# 1. Load Both Models
# ---------------------------------------------
CNN_MODEL_PATH = "models/cnn_model1_final.h5"
RESNET_MODEL_PATH = "models/resnet_emotion_model.h5"  # <-- Rename if needed

print("Loading CNN Model...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)

print("Loading ResNet Model...")
resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH, compile=False)

# Emotion labels (7-class FER2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------------------------------------
# 2. Preprocessing Function
# ---------------------------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 48x48 (model input size)
    img = cv2.resize(img, (48, 48))

    # Normalize
    img = img.astype("float32") / 255.0

    # Expand dims → shape (1, 48, 48, 3)
    return np.expand_dims(img, axis=0)

# ---------------------------------------------
# 3. Single Model Prediction Helper
# ---------------------------------------------
def predict_model(model, img):
    preds = model.predict(img)
    idx = np.argmax(preds)
    return {
        "predicted_class": emotion_labels[idx],
        "confidence": float(preds[0][idx]),
        "probabilities": preds[0].tolist()
    }

# ---------------------------------------------
# 4. Compare CNN vs ResNet
# ---------------------------------------------
def compare_models(img_path):
    img = preprocess_image(img_path)

    print("\nRunning CNN Model Prediction...")
    cnn_result = predict_model(cnn_model, img)

    print("Running ResNet Model Prediction...")
    resnet_result = predict_model(resnet_model, img)

    # Combined comparison result
    comparison = {
        "input_image": img_path,
        "cnn_prediction": cnn_result,
        "resnet_prediction": resnet_result
    }

    return comparison

# ---------------------------------------------
# 5. Run script directly
# ---------------------------------------------
if __name__ == "__main__":
    test_image = "test_image.jpg"  # <-- Replace with your own image
    
    result = compare_models(test_image)

    print("\n=========== MODEL COMPARISON ===========")
    print(f"Input Image: {result['input_image']}\n")

    print("----- CNN Model -----")
    print(f"Prediction: {result['cnn_prediction']['predicted_class']}")
    print(f"Confidence: {result['cnn_prediction']['confidence']:.4f}")

    print("\n----- ResNet Model -----")
    print(f"Prediction: {result['resnet_prediction']['predicted_class']}")
    print(f"Confidence: {result['resnet_prediction']['confidence']:.4f}")

    print("\n=========================================")
