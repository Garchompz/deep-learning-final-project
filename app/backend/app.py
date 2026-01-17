from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Updated to match training: EfficientNetV2B0
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import base64
import pickle
import os

app = Flask(__name__)
CORS(app)

# ======================================================
# CONSTANTS (Synchronized with Training)
# ======================================================
IMAGE_SIZE = (224, 224) # Training used 224 for EffNetV2B0
MAX_LEN = 35           # Training calculated max_length from Flickr8k
EMBEDDING_DIM = 512
LSTM_UNITS = 512       # Training used 512 units

# ======================================================
# CUSTOM LAYER: BAHDANAU ATTENTION
# ======================================================
@keras.utils.register_keras_serializable(package="Custom")
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)

    def call(self, inputs):
        # Training logic: context_vector, att_weights = attention([encoder_input, lstm_out])
        encoder_features, decoder_hidden = inputs
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)
        score = tf.nn.tanh(self.W1(encoder_features) + self.W2(decoder_hidden))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = tf.reduce_sum(attention_weights * encoder_features, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# ======================================================
# MODEL RECONSTRUCTION
# ======================================================
def reconstruct_caption_model(vocab_size):
    print("🔨 Reconstructing model architecture to match training...")
    
    # Encoder Output Shape for EffNetV2B0 is typically (batch, 7, 7, 1280) -> 49 patches
    encoder_input = layers.Input(shape=(49, 1280), name='image_features')
    decoder_input = layers.Input(shape=(MAX_LEN,), name='decoder_input')
    
    # Embedding (Matches Training)
    embed = layers.Embedding(input_dim=vocab_size, output_dim=512, mask_zero=True, name='embed')(decoder_input)
    embed = layers.Dropout(0.3)(embed)
    
    # Single LSTM (Training did NOT use Bidirectional)
    lstm_out = layers.LSTM(LSTM_UNITS, return_sequences=False, dropout=0.3, recurrent_dropout=0.2, name='lstm')(embed)
    lstm_out = layers.LayerNormalization()(lstm_out)
    
    # Attention
    attention_layer = BahdanauAttention(512)
    context_vector, _ = attention_layer([encoder_input, lstm_out])
    
    # Head (Matches Training logic)
    concat = layers.Concatenate(axis=-1)([context_vector, lstm_out])
    x = layers.Dense(512, activation='relu')(concat)
    x = layers.Dropout(0.4)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(x)
    
    return keras.Model(inputs=[encoder_input, decoder_input], outputs=outputs)

# ======================================================
# INITIALIZATION
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREV_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
TOKENIZER_PATH = os.path.join(PREV_DIR, "tokenizer.pkl")
MODEL_PATH = os.path.join(PREV_DIR, "caption_model_effnetv2b0.keras")

# Load Tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
WORD_INDEX = tokenizer.word_index
INDEX_WORD = tokenizer.index_word
VOCAB_SIZE = len(WORD_INDEX) + 1

# Load Model
try:
    # Attempting reconstruction first to avoid naming/versioning conflicts
    model = reconstruct_caption_model(VOCAB_SIZE)
    model.load_weights(MODEL_PATH)
    print("✅ Model weights loaded successfully!")
except Exception as e:
    print(f"❌ Critical Load Error: {e}")
    # Final fallback attempt
    model = keras.models.load_model(MODEL_PATH, custom_objects={'BahdanauAttention': BahdanauAttention}, compile=False)

# Load Encoder (EfficientNetV2B0)
print("🏗️ Initializing EfficientNetV2B0 Encoder...")
base_fe = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
encoder_model = keras.Model(inputs=base_fe.input, outputs=base_fe.output)

# ======================================================
# INFERENCE HELPERS
# ======================================================
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return preprocess_input(img_array) # EffNetV2 preprocessing

def extract_features(img_array):
    feat_map = encoder_model.predict(img_array, verbose=0)[0]
    h, w, c = feat_map.shape
    return feat_map.reshape(1, h * w, c) # Returns (1, 49, 1280)

def generate_caption(feature_vector, beam_index=5):
    start_seq = WORD_INDEX.get('startseq')
    end_seq = WORD_INDEX.get('endseq')
    
    sequences = [[[start_seq], 0.0]]
    
    for _ in range(MAX_LEN):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == end_seq:
                all_candidates.append([seq, score])
                continue
            
            padded = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
            preds = model.predict([feature_vector, padded], verbose=0)[0]
            
            top_indices = np.argsort(preds)[-beam_index:]
            for idx in top_indices:
                all_candidates.append([seq + [int(idx)], score - np.log(preds[idx] + 1e-10)])
        
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_index]
        if all(s[0][-1] == end_seq for s in sequences): break

    best_seq = sequences[0][0]
    words = [INDEX_WORD.get(i) for i in best_seq if i not in [start_seq, end_seq]]
    return " ".join(filter(None, words)).capitalize() + "."

# ======================================================
# ROUTES
# ======================================================
@app.route('/')
def index():
    return jsonify({"status": "online", "message": "Captioning Model API is running"})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file'].read()
        else:
            data = request.get_json()
            file = base64.b64decode(data['image'].split(',')[-1])

        processed_img = preprocess_image(file)
        features = extract_features(processed_img)
        caption = generate_caption(features)
        
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=7860)