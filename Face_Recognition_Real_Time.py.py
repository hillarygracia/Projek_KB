import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def get_images_and_labels(main_path='D:\\Tugas_Kampus\\Jurusan\\Kecerdasan_Buatan\\UAS\\dataset'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []
    label_names = {}
    current_label = 0

    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                img = cv2.imread(image_path)
                if img is None:
                    continue  
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

                for (x, y, w, h) in faces_detected:
                    faces.append(cv2.resize(gray[y:y+h, x:x+w], (150, 150)))
                    labels.append(current_label)

            current_label += 1

    return faces, labels, label_names

faces, labels, label_names = get_images_and_labels()

if len(faces) > 0:
    faces_flattened = [face.flatten() for face in faces]

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(faces_flattened, labels_encoded)

    np.save('svm_model.npy', clf)
    np.save('label_encoder.npy', le)
else:
    print("Dataset kosong atau tidak valid. Pastikan dataset berisi gambar wajah.")
    exit()

clf = np.load('svm_model.npy', allow_pickle=True).item()
le = np.load('label_encoder.npy', allow_pickle=True).item()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat mengakses webcam. Pastikan webcam terhubung.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        face_region_flattened = cv2.resize(gray[y:y+h, x:x+w], (150, 150)).flatten().reshape(1, -1)
        label_encoded = clf.predict(face_region_flattened)
        proba = clf.predict_proba(face_region_flattened)
        confidence = np.max(proba)  
        label = le.inverse_transform(label_encoded)[0]
        name = label_names.get(label, "Unknown")

        
        cv2.putText(frame, f"{name} ({int(confidence * 100)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()