import tkinter as tk
import numpy as np
# --- FIX START ---
from tensorflow.keras.models import load_model 
# --- FIX END ---
import pickle
from preprocess import resample_stroke

class RealTimeRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Character Recognition AI")
        
        # --- FIX START ---
        # Use the imported 'load_model' function directly
        self.model = load_model("models/dynamic_char_model.h5")
        # --- FIX END ---
        
        with open("models/label_map.pkl", "rb") as f:
            self.label_map = pickle.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # UI Setup
        self.canvas = tk.Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack()
        self.label_result = tk.Label(root, text="Draw a character...", font=("Helvetica", 20))
        self.label_result.pack()
        
        tk.Button(root, text="Clear", command=self.clear_canvas).pack()
        
        self.stroke = []
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.predict)

    def paint(self, event):
        x, y = event.x, event.y
        self.stroke.append([x, y])
        r = 4
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="blue")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.stroke = []
        self.label_result.config(text="Draw a character...")

    def predict(self, event):
        if len(self.stroke) < 5: return
        
        # 1. Preprocess
        raw = np.array(self.stroke)
        processed = resample_stroke(raw)
        
        # Normalize
        processed -= np.mean(processed, axis=0)
        max_val = np.abs(processed).max()
        if max_val > 0: processed /= max_val
        
        # 2. Reshape for LSTM (1, 50, 2)
        input_data = processed.reshape(1, 50, 2)
        
        # 3. Predict
        pred = self.model.predict(input_data)
        class_idx = np.argmax(pred)
        confidence = np.max(pred)
        
        char = self.inv_label_map[class_idx]
        self.label_result.config(text=f"Prediction: {char} ({confidence*100:.1f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeRecognizer(root)
    root.mainloop()