import pickle
from preprocess import load_data
from model import build_model
import os

# 1. Load Data
(X_train, X_test, y_train, y_test), label_map = load_data()

# 2. Build Model
input_shape = (X_train.shape[1], X_train.shape[2]) 
num_classes = len(label_map)

print(f"Training on {num_classes} classes...")
model = build_model(input_shape, num_classes)

# 3. Train
history = model.fit(X_train, y_train, 
                    epochs=60, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))

# 4. Save
os.makedirs("models", exist_ok=True)
model.save("models/dynamic_char_model.h5")

with open("models/label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Model trained and saved!")