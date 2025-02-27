import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import datetime
import threading
import pickle


with open('model.p', 'rb') as f:
    camera_model_dict = pickle.load(f)
camera_model = camera_model_dict['model']

camera_labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

uploaded_model = load_model('Sign_language_detector.h5')
uploaded_classes = np.load('classes.npy', allow_pickle=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.4)

def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros(42)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            return np.array(landmarks)
    return np.zeros(42)

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = preprocess_input(image)
    return image

def valid_time():
    current_time = datetime.datetime.now().time()
    start_time = datetime.time(22, 0)  # 10:00 PM
    end_time = datetime.time(15, 0)  # 1:00 AM (next day)
    if start_time <= current_time or current_time <= end_time:
        return True
    return False

def predict_on_uploaded_image(image_path):
    img = load_and_preprocess_image(image_path, img_size = (224, 224))
    img = np.expand_dims(img, axis = 0)
    
    lm = extract_hand_landmarks(image_path)
    lm = np.expand_dims(lm, axis = 0)
    
    prediction = uploaded_model.predict({'image_input': img, 'landmark_input': lm})
    predicted_index = np.argmax(prediction)
    predicted_character = uploaded_classes[predicted_index]
    return predicted_character

def process_camera_frame(frame):
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    prediction_text = "No hand detected"
    if valid_time() and results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        first_hand = results.multi_hand_landmarks[0]
        for lm in first_hand.landmark:
            x_.append(lm.x)
            y_.append(lm.y)
        for lm in first_hand.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))
        if data_aux:
            prediction = camera_model.predict([np.array(data_aux)])
            predicted_character = camera_labels_dict.get(int(prediction[0]), "Unknown")
            prediction_text = predicted_character
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    else:
        if not valid_time():
            prediction_text = "Application is accessible only between "
    cv2.putText(frame, prediction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0, 0, 0), 3, cv2.LINE_AA)
    return frame

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection")
        self.root.configure(bg="#f0f0f0")  
        
        top_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        self.upload_btn = tk.Button(top_frame, text="Upload Image", command=self.upload_image,
                                    font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.camera_btn = tk.Button(top_frame, text="Start Camera", command=self.start_camera,
                                    font=("Helvetica", 14), bg="#2196F3", fg="white", padx=10, pady=5)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        self.result_label = tk.Label(top_frame, text="", font=("Helvetica", 16), bg="#f0f0f0", fg="black")
        self.result_label.pack(side= tk.LEFT, padx=10)
   
        video_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.cap = None
        self.running = False

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            prediction = predict_on_uploaded_image(file_path)
            self.result_label.config(text=f"Prediction: {prediction}")
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.putText(img, f"{prediction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 0, 0), 3, cv2.LINE_AA)
                img_pil = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

    def start_camera(self):
        if self.cap is not None and self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")
            return
        self.running = True
        self.update_camera()

    def update_camera(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                processed_frame = process_camera_frame(frame)
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            self.root.after(10, self.update_camera)
        else:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    if app.cap is not None:
        app.cap.release()

if __name__ == "__main__":
    main()

