import os
import cv2
import numpy as np
import tensorflow as tf

import os as _os

# ── Paths ─────────────────────────────────────────────────────────────────────
# DIR_GENUINE / DIR_FORGED → your Kaggle signature image folders
# MODEL_SAVE_PATH          → saved into backend/models/ automatically
# Change DIR_GENUINE / DIR_FORGED if your folders are somewhere else.

DIR_GENUINE = r"D:\cyber\kaggle\signatures\full_org"
DIR_FORGED  = r"D:\cyber\kaggle\signatures\full_forg"

_HERE           = _os.path.dirname(_os.path.abspath(__file__))
MODEL_SAVE_PATH = _os.path.join(_HERE, "..", "models", "signature_siamese.keras")

def preprocess_image(path, target_size=(105, 105)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def load_dataset_pairs(gen_dir, forg_dir):
    left_images, right_images, labels = [], [], []
    
    gen_files = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    forg_files = sorted([os.path.join(forg_dir, f) for f in os.listdir(forg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Found {len(gen_files)} genuine files and {len(forg_files)} forged files.")

    # 1. TRUE POSITIVES: Pair identical or sequential signatures from the SAME individual
    # (Assumes sequential naming like user1_1.png, user1_2.png match)
    for i in range(len(gen_files) - 1):
        # Only pair as genuine if they belong to the same prefix/person identity
        name1 = os.path.basename(gen_files[i]).split('_')[0]
        name2 = os.path.basename(gen_files[i+1]).split('_')[0]
        
        if name1 == name2: 
            img1 = preprocess_image(gen_files[i])
            img2 = preprocess_image(gen_files[i+1])
            if img1 is not None and img2 is not None:
                left_images.append(img1)
                right_images.append(img2)
                labels.append(0.0) # 0.0 means Genuine Match

    # 2. INTERNAL NEGATIVES: Pair different people's genuine signatures together!
    # This prevents the model from assuming "all images in full_org are identical"
    for i in range(len(gen_files)):
        for j in range(i + 1, min(i + 10, len(gen_files))): # Check next few signatures
            name1 = os.path.basename(gen_files[i]).split('_')[0]
            name2 = os.path.basename(gen_files[j]).split('_')[0]
            
            if name1 != name2: # Completely different people
                img1 = preprocess_image(gen_files[i])
                img2 = preprocess_image(gen_files[j])
                if img1 is not None and img2 is not None:
                    left_images.append(img1)
                    right_images.append(img2)
                    labels.append(1.0) # 1.0 means Forged/Mismatched

    # 3. EXTERNAL NEGATIVES: Original genuine vs forged comparisons
    min_len = min(len(gen_files), len(forg_files))
    for i in range(min_len):
        img1 = preprocess_image(gen_files[i])
        img2 = preprocess_image(forg_files[i])
        if img1 is not None and img2 is not None:
            left_images.append(img1)
            right_images.append(img2)
            labels.append(1.0) 

    return np.array(left_images), np.array(right_images), np.array(labels, dtype='float32')

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.math.reduce_mean(y_true * margin_square + (1 - y_true) * square_pred)

@tf.keras.utils.register_keras_serializable()
class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        feats_a, feats_b = inputs
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(feats_a - feats_b), axis=1, keepdims=True))
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

def train():
    X_left, X_right, Y = load_dataset_pairs(DIR_GENUINE, DIR_FORGED)
    if len(Y) == 0:
        print("Error: No pairs could be generated.")
        return

    print(f"Compiled {len(Y)} varied training pairs. Training deep metrics...")

    input_shape = (105, 105, 1)
    base_net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu')
    ])

    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    feat_a = base_net(input_a)
    feat_b = base_net(input_b)

    distance = DistanceLayer()([feat_a, feat_b])

    model = tf.keras.Model(inputs=[input_a, input_b], outputs=distance)
    model.compile(optimizer='adam', loss=contrastive_loss)

    # Increased epochs slightly so the model learns the harder differentiations
    model.fit([X_left, X_right], Y, batch_size=8, epochs=12, validation_split=0.1)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"trained model saved cleanly to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()