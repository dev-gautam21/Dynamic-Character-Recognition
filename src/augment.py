import numpy as np
import os
import random

# CONFIGURATION
DATA_DIR = "data/raw"
AUGMENT_FACTOR = 50  # Generates 50 variations per drawing
NOISE_LEVEL = 2.0    
SCALE_RANGE = 0.15   
ROTATION_RANGE = 15  

def augment_stroke(stroke):
    # CRITICAL FIX: Convert to float for noise addition
    stroke = stroke.copy().astype(np.float64)
    
    # 1. Noise
    noise = np.random.normal(0, NOISE_LEVEL, stroke.shape)
    stroke += noise
    
    # 2. Scaling
    scale_x = 1 + random.uniform(-SCALE_RANGE, SCALE_RANGE)
    scale_y = 1 + random.uniform(-SCALE_RANGE, SCALE_RANGE)
    stroke[:, 0] *= scale_x
    stroke[:, 1] *= scale_y
    
    # 3. Rotation
    theta = np.radians(random.uniform(-ROTATION_RANGE, ROTATION_RANGE))
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    
    center = np.mean(stroke, axis=0)
    stroke -= center
    stroke = np.dot(stroke, rotation_matrix)
    stroke += center

    return stroke

def run_augmentation():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy') and "_aug" not in f]
    
    if not files:
        print("No data found! Run collect_data.py first.")
        return

    print(f"Found {len(files)} original files. Generating {AUGMENT_FACTOR} variations each...")

    count = 0
    for filename in files:
        try:
            file_path = os.path.join(DATA_DIR, filename)
            original_stroke = np.load(file_path)
            
            # Extract label (e.g. "A" from "A_123.npy")
            label_part = filename.split('_')[0]
            
            for i in range(AUGMENT_FACTOR):
                new_stroke = augment_stroke(original_stroke)
                new_filename = f"{label_part}_aug_{i}_{filename}"
                np.save(os.path.join(DATA_DIR, new_filename), new_stroke)
                count += 1
                
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"Success! Created {count} new training samples.")

if __name__ == "__main__":
    run_augmentation()