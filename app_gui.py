import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# --- Main Application Class using CustomTkinter ---
class TomatoDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Disease Detector")
        self.root.geometry("400x550")
        
        # Set the appearance mode
        ctk.set_appearance_mode("light") 
        ctk.set_default_color_theme("green")

        # --- Load the Model ---
        print("Loading the trained model...")
        self.model = tf.keras.models.load_model('tomato_disease_model.h5')
        print("Model loaded successfully!")

        # --- Define Class Names ---
        self.class_names = [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot',
            'Tomato_YellowLeaf_Curl_Virus', 'Tomato_mosaic_virus', 'Tomato_healthy'
        ]

        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.main_frame.pack(padx=20, pady=20, expand=True, fill='both')

        # --- Create GUI Widgets ---
        self.create_widgets()

    def create_widgets(self):
        # --- Title ---
        title_label = ctk.CTkLabel(self.main_frame, text="Tomato Disease Detector", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(20, 10))
        
        subtitle_label = ctk.CTkLabel(self.main_frame, text="Upload an image for an AI-powered diagnosis.", font=ctk.CTkFont(size=12), text_color="gray50")
        subtitle_label.pack(pady=(0, 20))

        # --- Image Display Area ---
        self.image_label = ctk.CTkLabel(self.main_frame, text="Click below to select an image", width=300, height=200, fg_color="gray85", corner_radius=10, text_color="gray50")
        self.image_label.pack()

        # --- Select Image Button ---
        self.select_button = ctk.CTkButton(self.main_frame, text="Select Image", font=ctk.CTkFont(size=14), command=self.select_image)
        self.select_button.pack(pady=20)

        # --- Result Labels ---
        self.result_label = ctk.CTkLabel(self.main_frame, text="", font=ctk.CTkFont(size=18, weight="bold"))
        self.result_label.pack()
        
        self.confidence_label = ctk.CTkLabel(self.main_frame, text="", font=ctk.CTkFont(size=12))
        self.confidence_label.pack(pady=(0, 20))
        
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        # --- Display the selected image ---
        img = Image.open(file_path)
        ctk_img = ctk.CTkImage(light_image=img, size=(300, 200))
        self.image_label.configure(image=ctk_img, text="")

        # --- Make a prediction ---
        self.predict(file_path)

    def predict(self, file_path):
        # Preprocess the image
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Get the prediction
        prediction = self.model.predict(img_array)
        
        # Decode the prediction
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = self.class_names[predicted_class_index].replace('_', ' ')
        confidence = float(np.max(prediction[0]))

        # --- Update the result labels ---
        self.result_label.configure(text=predicted_class_name)
        self.confidence_label.configure(text=f"Confidence: {confidence:.2%}")

# --- Main entry point to run the application ---
if __name__ == "__main__":
    root = ctk.CTk()
    app = TomatoDetectorApp(root)
    root.mainloop()

