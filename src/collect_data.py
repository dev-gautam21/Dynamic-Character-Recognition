import tkinter as tk
import numpy as np
import os
import time

# CONFIGURATION
DATA_PATH = "data/raw"
os.makedirs(DATA_PATH, exist_ok=True)

# A-Z and 0-9
CHAR_LIST = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

class RapidCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Data Collector")
        
        self.index = 0
        self.current_char = CHAR_LIST[self.index]
        
        # --- UI SETUP ---
        # Large Label for the current character
        self.info_label = tk.Label(root, text=f"Draw: {self.current_char}", font=("Helvetica", 60, "bold"), fg="#333")
        self.info_label.pack(pady=10)
        
        self.hint_label = tk.Label(root, text="Draw the full character, then click NEXT", font=("Helvetica", 14), fg="blue")
        self.hint_label.pack()
        
        self.progress_label = tk.Label(root, text=f"Progress: {self.index + 1}/{len(CHAR_LIST)}")
        self.progress_label.pack()

        # Drawing Canvas
        self.canvas = tk.Canvas(root, width=300, height=300, bg="white", highlightthickness=4, highlightbackground="#ccc")
        self.canvas.pack(pady=10)
        
        self.stroke = []
        
        # --- BINDINGS ---
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Keyboard Shortcut: Press SPACE to click "Next"
        root.bind("<space>", self.save_and_advance)
        
        # --- MAIN BUTTON ---
        tk.Button(root, text="NEXT (Save) >", command=self.save_and_advance, 
                  bg="#4CAF50", fg="black", font=("Helvetica", 16, "bold"), height=2, width=15).pack(pady=5)
        
        # Control Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Clear / Retry", command=self.clear_canvas, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Skip Letter", command=self.skip_char, height=2).pack(side=tk.LEFT, padx=10)

    def paint(self, event):
        x, y = event.x, event.y
        self.stroke.append([x, y])
        r = 4
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")

    def save_and_advance(self, event=None):
        """Called when Button is clicked or Spacebar is pressed"""
        # 1. Validation: Don't save empty canvas
        if len(self.stroke) < 5:
            self.hint_label.config(text="Canvas is empty! Draw first.", fg="red")
            return
            
        # 2. Save File
        filename = f"{DATA_PATH}/{self.current_char}_{int(time.time())}.npy"
        np.save(filename, np.array(self.stroke))
        print(f"Saved {filename}")
        
        # 3. Advance to Next
        self.index += 1
        if self.index >= len(CHAR_LIST):
            self.info_label.config(text="DONE!", fg="green")
            self.hint_label.config(text="All characters collected.")
            self.canvas.unbind("<B1-Motion>")
            return
            
        self.update_ui()

    def update_ui(self):
        self.current_char = CHAR_LIST[self.index]
        self.info_label.config(text=f"Draw: {self.current_char}")
        self.progress_label.config(text=f"Progress: {self.index + 1}/{len(CHAR_LIST)}")
        self.hint_label.config(text="Draw the full character, then click NEXT", fg="blue")
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.stroke = []

    def skip_char(self):
        self.index += 1
        if self.index < len(CHAR_LIST):
            self.update_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = RapidCollector(root)
    root.mainloop()