import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

FIXED_LENGTH = 50 

def resample_stroke(stroke, n=FIXED_LENGTH):
    if len(stroke) == 0: return np.zeros((n, 2))
    
    dist = np.cumsum(np.sqrt(np.sum(np.diff(stroke, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)
    total_dist = dist[-1]
    
    if total_dist == 0: return np.resize(stroke, (n, 2))

    new_dists = np.linspace(0, total_dist, n)
    new_x = np.interp(new_dists, dist, stroke[:, 0])
    new_y = np.interp(new_dists, dist, stroke[:, 1])
    
    return np.stack([new_x, new_y], axis=1)

def load_data(data_dir="data/raw"):
    sequences = []
    labels = []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # Get unique labels
    unique_labels = sorted(list(set([f.split('_')[0] for f in files])))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    print(f"Classes found ({len(unique_labels)}): {unique_labels}")

    for f in files:
        label_str = f.split('_')[0]
        raw_stroke = np.load(os.path.join(data_dir, f))
        
        # 1. Resample
        processed_stroke = resample_stroke(raw_stroke)
        
        # 2. Normalize
        processed_stroke -= np.mean(processed_stroke, axis=0)
        max_val = np.abs(processed_stroke).max()
        if max_val > 0: processed_stroke /= max_val
            
        sequences.append(processed_stroke)
        labels.append(label_map[label_str])
        
    X = np.array(sequences)
    y = to_categorical(labels)
    
    return train_test_split(X, y, test_size=0.2), label_map