import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import  load_model

class MNISTPredictorApp:
    def __init__(self, master):
        self.master = master
        self.master.title('MNIST Digit Predictor')

        self.model = load_model('./model.h5')

        self.canvas = tk.Canvas(self.master, width = 200, height = '200')
        self.canvas.pack()

        self.label_result = tk.Label(self.master, text = 'Prediction: ')
        self.label_result.pack(pady = 10)

        self.btn_browse = tk.Button(self.master, text = 'Browse Image', command = self.browse_image)
        self.btn_browse.pack(pady = 10)

        self.img = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes = [("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(img)
            self.canvas.config(width = self.img.width(), height = self.img.height())
            self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)

            img_array = self.preprocess_image(file_path)
            predicted_label = self.predict_digit(img_array)
            
            self.label_result.config(text = f"Prediction: {predicted_label}")
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.astype('float32') / 255.0
        img = np.reshape(img, (1, 28, 28, 1))
        return img

    def predict_digit(self, img_array):
        prediction = self.model.predict(img_array)
        predicted_label = np.argmax(prediction)
        return predicted_label


if __name__ == '__main__':
    root = tk.Tk()
    app = MNISTPredictorApp(root)
    root.mainloop()