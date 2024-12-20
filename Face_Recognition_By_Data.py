import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset_path = "dataset"

image_size = (250, 250)

def load_dataset(dataset_path):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, image_size)
                    images.append(img_resized.flatten())
                    labels.append(label)
    return np.array(images), np.array(labels), class_names

print("Loading dataset...")
X, y, class_names = load_dataset(dataset_path)

num_classes = len(np.unique(y))
test_size = max(2, num_classes)
if len(X) <= test_size:
    raise ValueError("Dataset nya terlalu sedikit, mohon perbanyak dataset yang ingin digunakan.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

print("Melatih Model SVM...")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

print("Memvalidasi Model...")
y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

def predict_image(image_path):
    results = []
    for path in image_path:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, image_size).flatten()
            prediction = svm_model.predict([img_resized])
            results.append(class_names[prediction[0]])
        else:
            results.append(None)
    return results

image_to_test = ["validator/ali/ali.jpg", "validator/ali/ali2.jpg"]
predicted_classes = predict_image(image_to_test)
for image, predicted_class in zip(image_to_test, predicted_classes):
    print(f"Image: {image}, Predicted class: {predicted_class}")