# Happy and Sad Faces Detection using CNN 🧠📸

This project is a Convolutional Neural Network (CNN)-based deep learning model to classify facial emotions into **Happy 😀** or **Sad 😞** categories.  
It uses TensorFlow/Keras and a Kaggle dataset to train and evaluate the model.

---

## 🚀 Features
- Custom CNN architecture with Conv2D, MaxPooling, Dense & Dropout layers  
- Data augmentation (rotation, zoom, shifts, flips) for robust training  
- Early stopping callback when accuracy ≥ 81%  
- Training and validation performance visualization  

---

## 📂 Dataset
Dataset used: [Happy or Sad Emotion Detection (Kaggle)](https://www.kaggle.com/datasets/aravindanr22052001/emotiondetection-happy-or-sad)

- Images classified into **Happy** and **Sad** categories  
- Automatically split into Training (80%) and Validation (20%)  

---

## 🛠️ Tech Stack
- Python 3  
- TensorFlow / Keras  
- Matplotlib for visualization  
- Google Colab / Kaggle for training  

---

## 📊 Model Performance
- Achieved **~82% accuracy** on training data  
- Validation accuracy stabilized around **70%**  
- Can be further improved with a larger dataset and hyperparameter tuning  

---

## ▶️ How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/nitinog10/Happy-and-sad-faces-detection-using-ml.git
   cd Happy-and-sad-faces-detection-using-ml
Install dependencies

pip install tensorflow matplotlib kagglehub


Run the notebook on Colab:
Happy and Sad Faces Detection Notebook

📸 Results

The model successfully detects emotions in face images and plots accuracy/loss trends:

✅ Training Accuracy reached above 80%

📉 Validation accuracy stabilized at 65–70%

📌 Future Scope

Add more emotion categories (Angry, Neutral, Surprise, etc.)

Use transfer learning (VGG16, ResNet50) for better accuracy

Deploy as a web app using Flask/Streamlit

👨‍💻 Author

Developed by Nitin Mishra
