import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from deepface import DeepFace
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Load CNN model
def load_cnn_model(model_path):
    return load_model(model_path)

# Preprocess image for CNN model
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Capture image using webcam
def capture_image():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Image", frame)
        
        k = cv2.waitKey(1)
        if k % 256 == 32:  # Press Space to capture
            img_name = "captured_image.jpg"
            cv2.imwrite(img_name, frame)
            print("Image captured!")
            break

    cam.release()
    cv2.destroyAllWindows()
    return img_name

# Check if a face is present
def is_face_present(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

# Analyze emotion using DeepFace
def analyze_emotion_deepface(image_path):
    if not is_face_present(image_path):
        print("No face detected. Try recapturing the image.")
        return "No face detected"

    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion'] if result else "No emotion detected"
    except Exception as e:
        print(f"DeepFace error: {e}")
        return "DeepFace error"

# Analyze emotion using CNN model
def analyze_emotion_cnn(image_path, model):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    predicted_emotion = emotions[np.argmax(predictions)]
    return predicted_emotion

# Perform sentiment analysis on text
def analyze_text_sentiment(text):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Main execution
if __name__ == "__main__":
    model_path = "emotion_cnn_model.h5"  # Ensure this file exists
    if os.path.exists(model_path):
        cnn_model = load_cnn_model(model_path)
    else:
        cnn_model = None
        print("CNN model not found. Using only DeepFace.")

    img_path = capture_image()
    detected_emotion = analyze_emotion_deepface(img_path)
    
    if detected_emotion == "No face detected" or detected_emotion == "DeepFace error":
        if cnn_model:
            print("Using CNN model as fallback.")
            detected_emotion = analyze_emotion_cnn(img_path, cnn_model)
    
    print(f"Detected Emotion: {detected_emotion}")

    text_input = input("Enter text for sentiment analysis: ")
    sentiment_result = analyze_text_sentiment(text_input)
    print("Sentiment Analysis:", sentiment_result)
