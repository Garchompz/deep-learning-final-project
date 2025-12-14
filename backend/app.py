from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io
import base64
import pickle

# Register custom layers
register_keras_serializable = keras.utils.register_keras_serializable

app = Flask(__name__)
CORS(app)

# ======================================================
# CONSTANTS
# ======================================================
IMAGE_SIZE = (260, 260)
MAX_LEN = 40
EMBEDDING_DIM = 256
UNITS = 256

# ======================================================
# CUSTOM LAYER: BAHDANAU ATTENTION (FIXED)
# ======================================================
@register_keras_serializable(package="Custom", name="BahdanauAttention")
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def build(self, input_shape):
        # Explicit build method to satisfy Keras loading mechanism
        # input_shape is usually a list of shapes [features_shape, hidden_shape]
        super(BahdanauAttention, self).build(input_shape)

    def call(self, inputs, hidden=None):
        # === FIX LOGIC FOR KERAS DESERIALIZATION ===
        # Keras sering mengirim inputs sebagai list [features, hidden] ke argumen pertama
        if hidden is None:
            features, hidden = inputs
        else:
            features = inputs
        # ===========================================

        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_weights shape == (batch_size, 64, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

# ======================================================
# MANUAL MODEL RECONSTRUCTION (BACKUP PLAN)
# ======================================================
def reconstruct_caption_model(vocab_size):
    """
    Membangun ulang arsitektur model secara manual agar 100% cocok dengan notebook.
    Ini digunakan jika load_model gagal melakukan deserialization layer custom.
    """
    print("🔨 Reconstructing model architecture manually...")
    
    # Encoder Input
    # Shape dari EffNetB2 (260x260) -> output map 8x8 (approx) -> 64 patches
    # Channel EffNetB2 = 1408
    # Kita gunakan None untuk num_patches agar fleksibel
    encoder_input = layers.Input(shape=(None, 1408), name='image_features')
    
    # Decoder Input
    decoder_input = layers.Input(shape=(MAX_LEN,), name='decoder_input')
    
    # Embedding
    embed = layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name='embed'
    )(decoder_input)
    
    # LSTM 1
    bilstm_1 = layers.Bidirectional(
        layers.LSTM(UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        name='bilstm_1'
    )(embed)
    
    # LSTM 2
    bilstm_2 = layers.Bidirectional(
        layers.LSTM(UNITS, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
        name='bilstm_2'
    )(bilstm_1)
    
    # Attention
    attention_layer = BahdanauAttention(UNITS)
    # Ini cara panggil di training: attention(encoder_input, bilstm_2)
    context_vector, att_weights = attention_layer([encoder_input, bilstm_2])
    
    # Concatenate
    concat = layers.Concatenate(axis=-1)([context_vector, bilstm_2])
    
    # Dense Layers
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=[encoder_input, decoder_input], outputs=outputs)
    return model

# ======================================================
# 1. LOAD TOKENIZER FIRST (Need vocab size for model)
# ======================================================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    
    if hasattr(tokenizer, 'index_word'):
        INDEX_WORD = tokenizer.index_word
        WORD_INDEX = tokenizer.word_index
    else:
        INDEX_WORD = tokenizer['index_word']
        WORD_INDEX = tokenizer['word_index']
        
    VOCAB_SIZE = len(WORD_INDEX) + 1
    print(f"✅ Tokenizer loaded. Vocab size: {VOCAB_SIZE}")
    
except Exception as e:
    print(f"⚠️ Tokenizer loading failed: {e}")
    # Fallback default untuk mencegah crash sebelum load model
    VOCAB_SIZE = 15000 
    INDEX_WORD = {}
    WORD_INDEX = {}

# ======================================================
# 2. LOAD MODEL (With Fail-Safe Strategy)
# ======================================================

custom_objects = {'BahdanauAttention': BahdanauAttention, "TransformerDecoderLayer": TransformerDecoderLayer}
model = None
MODEL_PATH = os.path.join(BASE_DIR, "caption_trained_model_effb2.keras")

print("=" * 60)
print("🔄 Attempting to load Model...")

# CARA 1: Load Full Model (Preferred)
try:
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ Model loaded successfully via load_model!")

except Exception as e:
    print(f"⚠️ load_model failed: {e}")
    print("🔄 Switching to Strategy 2: Reconstruct Architecture + Load Weights")
    
    # CARA 2: Reconstruct + Load Weights (Fail-Safe)
    try:
        model = reconstruct_caption_model(VOCAB_SIZE)
        # Load weights only
        model.load_weights(MODEL_PATH)
        print("✅ Model reconstructed and weights loaded successfully!")
    except Exception as e2:
        print(f"❌ CRITICAL: Could not load model even after reconstruction: {e2}")
        raise e2

# ======================================================
# 3. LOAD ENCODER (EfficientNetB2)
# ======================================================

print("🏗️ Initializing EfficientNetB2 Encoder...")
encoder_base = EfficientNetB2(
    include_top=False, 
    weights='imagenet', 
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
encoder_model = keras.Model(inputs=encoder_base.input, outputs=encoder_base.output)
print("✅ Encoder initialized")

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(img_array):
    features = encoder_model.predict(img_array, verbose=0)
    batch, h, w, c = features.shape
    # Reshape (Batch, 8*8, 1408) -> (Batch, 64, 1408)
    features_reshaped = features.reshape(batch, h * w, c) 
    return features_reshaped

def clean_caption(caption):
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    return caption.capitalize()

def generate_caption_beam(feature_vector, beam_index=3):
    start_seq = WORD_INDEX.get('startseq')
    end_seq = WORD_INDEX.get('endseq')
    
    if not start_seq: return "Error: Tokenizer missing 'startseq'"
    
    sequences = [[[start_seq], 0.0]]
    
    for _ in range(MAX_LEN):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == end_seq:
                all_candidates.append([seq, score])
                continue
            
            seq_pad = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
            
            # Predict
            # INPUT: [image_features, decoder_sequence]
            yhat = model.predict([feature_vector, seq_pad], verbose=0)
            probs = yhat[0]
            
            top_indices = np.argsort(probs)[-beam_index:]
            
            for word_idx in top_indices:
                p = probs[word_idx]
                if p <= 1e-9: continue
                new_score = score + np.log(p)
                new_seq = seq + [int(word_idx)]
                all_candidates.append([new_seq, new_score])
        
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_index]
        
        if sequences[0][0][-1] == end_seq and len(sequences[0][0]) > 1:
            break

    best_seq = sequences[0][0]
    final_caption = [INDEX_WORD.get(idx) for idx in best_seq if idx not in [start_seq, end_seq] and INDEX_WORD.get(idx)]
    return clean_caption(" ".join(final_caption))

# ======================================================
# ROUTES
# ======================================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_bytes = None
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json(force=True)
            img_b64 = data.get('image') or data.get('data')
            if not img_b64: return jsonify({'error': 'No image data'}), 400
            if ',' in img_b64: img_b64 = img_b64.split(',', 1)[1]
            img_bytes = base64.b64decode(img_b64)
        elif 'file' in request.files:
            img_bytes = request.files['file'].read()
        else:
            return jsonify({'error': 'No file uploaded'}), 400

        img_array = preprocess_image(img_bytes)
        features = extract_features(img_array)
        caption = generate_caption_beam(features)

        return jsonify({'caption': caption})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == "__main__":
    print("\n🚀 Starting Flask Server...")
    app.run(debug=True, host='0.0.0.0', port=5000)