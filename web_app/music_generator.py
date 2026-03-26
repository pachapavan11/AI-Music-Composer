import os
import pickle
import numpy as np
import tensorflow as tf
from music21 import stream, tempo, instrument, chord, note
# Try to import optional audio libraries
try:
    import pretty_midi as pm
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio generation libraries (pretty_midi, soundfile) not found. WAV generation will be skipped.")

class ModelWrapper:
    def __init__(self, model, unique_tokens, token_to_int, int_to_token):
        self.model = model
        self.unique_tokens = unique_tokens
        self.token_to_int = token_to_int
        self.int_to_token = int_to_token

        # Optimization: Create a specific predict function
        @tf.function
        def predict_fn(x):
            return self.model(x, training=False)
        self._predict_fn = predict_fn

    def predict_logits(self, arr):
        x = tf.constant(arr.astype(np.int32))
        return self._predict_fn(x).numpy()

def load_models_data(genre_to_model_map, genre_to_mapping_map):
    loaded = {}
    for genre, model_path in genre_to_model_map.items():
        try:
            print(f"Loading model for {genre} from {model_path}...")
            # Load model
            # Note: We need to define AttentionLayer if it's a custom layer in the saved model
            # defined in app.py or we define it here if needed for loading Keras model
            from tensorflow.keras import layers
            class AttentionLayer(layers.Layer):
                def __init__(self, units, **kwargs):
                    super(AttentionLayer, self).__init__(**kwargs)
                    self.units = units
                    self.W1 = layers.Dense(units)
                    self.W2 = layers.Dense(units)
                    self.V = layers.Dense(1)
                def call(self, values, query):
                    query_with_time = tf.expand_dims(query, 1)
                    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time)))
                    attention_weights = tf.nn.softmax(score, axis=1)
                    context_vector = attention_weights * values
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, attention_weights
                def get_config(self):
                    config = super().get_config()
                    config.update({"units": self.units})
                    return config

            model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
            
            # Load mapping
            mapping_path = genre_to_mapping_map[genre]
            print(f"Loading mapping for {genre} from {mapping_path}...")
            with open(mapping_path, "rb") as f:
                # The pickle structure from previous analysis was (vocab, token_to_int, int_to_token)
                unique_tokens, token_to_int, int_to_token = pickle.load(f)
            
            loaded[genre] = ModelWrapper(model, unique_tokens, token_to_int, int_to_token)
            print(f"Successfully loaded {genre}.")
        except Exception as e:
            print(f"Error loading model for {genre}: {e}")
    return loaded

# =========================================
# GENERATION HELPERS
# =========================================
def sample_from_probs(probs, temp):
    probs = np.asarray(probs)
    # Avoid div by zero
    logits = np.log(probs + 1e-12) / temp
    logits -= np.max(logits)
    dist = np.exp(logits)
    dist /= np.sum(dist)
    return np.random.choice(len(dist), p=dist)

def build_seed(unique, t2i, seq_len):
    # Filter out REST tokens for the very first seed note if possible, to start with sound
    valid = [t for t in unique if t and not t.startswith("REST")]
    if not valid: valid = unique
    token = np.random.choice(valid)
    return [t2i[token]] * seq_len

def generate_tokens(wrapper, seq_len, length=None, temp=1.0, max_duration_beats=None):
    """
    Generate tokens.
    If max_duration_beats is provided, stops when total duration exceeds it.
    If length is provided, generates that many tokens.
    """
    pattern = build_seed(wrapper.unique_tokens, wrapper.token_to_int, seq_len)
    output = []
    
    current_beats = 0.0

    # Duration mapping estimate
    dur_map = {'16': 0.25, '8': 0.5, 'q': 1.0, 'h': 2.0, 'w': 4.0}

    print(f"Generating music...")
    
    # Safety limit
    max_steps = 1000 
    if length: max_steps = length
    elif max_duration_beats: max_steps = int(max_duration_beats * 4 * 1.5) # generous buffer
    
    for i in range(max_steps):
        # Check duration condition
        if max_duration_beats and current_beats >= max_duration_beats:
            break
            
        arr = np.array([pattern[-seq_len:]], dtype=np.int32)
        pred = wrapper.predict_logits(arr)[0]

        probs = np.exp(pred - np.max(pred))
        probs /= np.sum(probs)

        idx = sample_from_probs(probs, temp)
        token = wrapper.int_to_token.get(idx, None)

        if token:
            output.append(token)
            pattern.append(idx)
            
            # Estimate duration added
            if '_' in token:
                 _, dur_char = token.rsplit('_', 1)
                 current_beats += dur_map.get(dur_char, 0.25)
            else:
                 current_beats += 0.25 # default assumption
        else:
            pattern.append(0)

    print(f"Generated {len(output)} tokens, approx {current_beats:.2f} beats.")
    return output

def tokens_to_midi(tokens, out, bpm, note_len_multiplier=1.0):
    # note_len_multiplier is roughly how many quarter notes a 'q' token represents if we want scaling
    # But detailed logic: token like C4_q means Quarter note. 
    # The user passed note_len from config. Let's use it as a base scale or tempo modifier if needed.
    # Actually, the user's snippet uses note_len as quarterLength assignment for everything?
    # "n.quarterLength = note_len". This overrides the token's intrinsic duration (e.g. _16, _w).
    # IF the token string has duration (e.g. C4_16), we should probably respect it, 
    # BUT the user's provided logic sets everything to `note_len`. 
    # I will adapt to use the token's duration if present, OR user's note_len if intended to override.
    # Looking at user code: 
    # if "." in t: ch.quarterLength = note_len
    # else: n.quarterLength = note_len
    # This implies a fixed duration per token generation step, ignoring the token's suffix?
    # Wait, the tokens are like "C4_q". 
    # If the model emits "C4_q", "C4_w", etc., setting all to fixed `note_len` (e.g. 0.8) makes the rhythm uniform.
    # Maybe that's what they want for "Creative" techno/pop? 
    # I will stick to the user's provided logic strictly initially.
    
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=bpm))
    s.append(instrument.Piano())
    offset = 0

    for t in tokens:
        if not t or t.startswith("REST"):
            offset += note_len_multiplier
            continue

        try:
            # Handle duration suffix if we want to be smarter? 
            # User code: simple split by "." for chords.
            # Token example: "C4.E4_q" or "C4_16".
            # User code ignores the "_q" suffix handling explicitly for duration?
            # It creates Note(t) which might fail if t is "C4_q". music21 Note("C4_q")? No.
            # music21 expects "C4". 
            # So we MUST strip the duration suffix.
            
            pitch_content = t
            if '_' in t:
                pitch_content, _ = t.rsplit('_', 1)
            
            if "." in pitch_content:
                ch = chord.Chord(pitch_content.split("."))
                ch.quarterLength = note_len_multiplier
                ch.offset = offset
                s.append(ch)
            else:
                n = note.Note(pitch_content)
                n.quarterLength = note_len_multiplier
                n.offset = offset
                s.append(n)
        except Exception as e:
            # print(f"Skipping token {t}: {e}")
            pass
        offset += note_len_multiplier

    os.makedirs(os.path.dirname(out), exist_ok=True)
    s.write("midi", fp=out)
    return out

def midi_to_wav(midi_path, wav_out, sample_rate=44100):
    if not AUDIO_AVAILABLE:
        return None
        
    try:
        # standard PrettyMIDI synthesis
        midi = pm.PrettyMIDI(midi_path)
        # synthesize() uses a simple sine wave synthesis if fluidsynth is not working or present?
        # actually synthesize() is usually sine waves. fluidsynth() needs soundfonts.
        # User code had two attempts: fluidsynth then synthesize.
        try:
            audio = midi.fluidsynth(fs=sample_rate)
        except:
            audio = midi.synthesize(fs=sample_rate)
            
        sf.write(wav_out, audio, sample_rate)
        return wav_out
    except Exception as e:
        print(f"WAV generation failed: {e}")
        return None
