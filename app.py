import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pickle
from music21 import instrument, note, chord, stream as m21_stream, tempo as m21_tempo

# Optional audio rendering
try:
    import pretty_midi as pm
    import soundfile as sf
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

# -------------------- Page Config & Theming --------------------
st.set_page_config(
    page_title="AI Music Composer",
    page_icon="🎵",
    layout="wide",
)

# Subtle modern styling
st.markdown(
    """
    <style>
    /* App background (target multiple containers to ensure it takes effect) */
    html, body { background: transparent !important; }
    .stApp { background: transparent; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0b0f14 0%, #0e141b 55%, #101821 100%) !important;
    }
    .main { /* legacy container */
        background: linear-gradient(180deg, #0b0f14 0%, #0e141b 55%, #101821 100%);
        color: #e6e6e6;
        font-size: 16px;
    }
    .stApp header {background: transparent;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {background: rgba(255,255,255,0.035); border-radius: 14px; padding: 14px; border: 1px solid rgba(255,255,255,0.08);}    
    .pill {display:inline-block; padding: 6px 12px; border-radius: 999px; background: rgba(120,180,255,0.10); font-size: 13px; margin-left: 8px;}
    .gradient-title {background: linear-gradient(90deg, #93c5fd 0%, #7dd3fc 55%, #a5b4fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 52px; line-height: 1.15;}
    .accent {color: #7dd3fc;}
    .stSlider > div > div > div > div { background: #7dd3fc; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #7dd3fc, #a5b4fc); }
    .download-wrap {display:flex; gap:10px; flex-wrap:wrap;}
    .stButton button { font-size: 16px; padding: 12px 18px; }

    /* Professional brand header */
    .brand-wrap { display:flex; flex-direction:column; align-items:center; gap:8px; }
    .brand-header { display:flex; align-items:center; justify-content:center; gap:14px; }
    .brand-icon {
        width: 42px; height: 42px; border-radius: 12px;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, rgba(125,211,252,0.25), rgba(165,180,252,0.25));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25), inset 0 0 0 1px rgba(255,255,255,0.06);
        backdrop-filter: blur(4px);
    }
    .brand-icon span { font-size: 24px; }
    .brand-title {
        background: linear-gradient(90deg, #b7cffd 0%, #d8c3ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; letter-spacing: 0.4px; margin: 0;
        font-size: clamp(34px, 5vw, 52px);
    }
    .brand-subtitle { opacity: 0.9; font-size: 17px; }

    /* Sidebar Controls title */
    .sidebar-title {
        font-weight: 900;
        font-size: 22px;
        letter-spacing: 0.6px;
        margin: 6px 0 10px 0;
        text-transform: uppercase;
    }

    /* Sidebar entrance + smooth interactions */
    section[data-testid="stSidebar"] {
        animation: sidebarEnter 420ms cubic-bezier(0.2, 0.9, 0.25, 1) both;
        will-change: transform, opacity;
        background: #121820 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    @keyframes sidebarEnter {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Group subtle hover/focus feedback */
    section[data-testid="stSidebar"] .sidebar-group {
        transition: background 220ms ease, transform 220ms ease, box-shadow 220ms ease;
        border-radius: 12px;
    }
    section[data-testid="stSidebar"] .sidebar-group:hover {
        background: rgba(255,255,255,0.03);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
    }

    /* Smooth, larger radios inside a boxed card */
    section[data-testid="stSidebar"] .stRadio {
        --accent: #7dd3fc;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18), inset 0 0 0 1px rgba(255,255,255,0.02);
        transition: background 220ms ease, box-shadow 220ms ease, transform 220ms ease, border-color 220ms ease;
    }
    section[data-testid="stSidebar"] .stRadio:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(125,211,252,0.25);
        box-shadow: 0 12px 28px rgba(0,0,0,0.22), 0 0 0 1px rgba(125,211,252,0.08) inset;
        transform: translateY(-1px);
    }
    /* Group label styled as a subtle header bar */
    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.3px;
        margin: 2px 2px 10px 2px;
        padding-bottom: 8px;
        border-bottom: 1px dashed rgba(255,255,255,0.12);
        color: #e6f1ff;
        text-transform: uppercase;
    }
    /* Radiogroup in two columns */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px 14px;
    }
    /* Option labels (inside radiogroup): subtle hover animation */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        font-size: 16px;
        transition: color 180ms ease, transform 160ms ease, text-shadow 160ms ease;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        color: #b7d6ff;
        transform: translateX(2px);
        text-shadow: 0 0 6px rgba(106,169,255,0.25);
    }
    /* BaseWeb radio icon sizing/animation */
    section[data-testid="stSidebar"] .stRadio svg {
        transition: transform 160ms ease, filter 220ms ease;
        filter: drop-shadow(0 0 0 transparent);
    }
    section[data-testid="stSidebar"] .stRadio [aria-checked="true"] svg {
        transform: scale(1.06);
        filter: drop-shadow(0 0 6px rgba(106,169,255,0.35));
    }

    /* Time slider: interactive thumb + gentle hover */
    section[data-testid="stSidebar"] .stSlider {
        padding: 6px 8px;
        border-radius: 12px;
        transition: background 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }
    section[data-testid="stSidebar"] .stSlider:hover {
        background: rgba(255,255,255,0.03);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
        transform: translateY(-1px);
    }
    section[data-testid="stSidebar"] .stSlider [role="slider"] {
        transition: transform 140ms ease, box-shadow 180ms ease;
        box-shadow: 0 0 0 0 rgba(106,169,255,0.0);
    }
    section[data-testid="stSidebar"] .stSlider [role="slider"]:hover {
        transform: scale(1.04);
        box-shadow: 0 0 0 6px rgba(106,169,255,0.12);
    }
    section[data-testid="stSidebar"] .stSlider [role="slider"][aria-valuenow] {
        outline: none !important;
    }

    /* Toggles: subtle hover feedback */
    section[data-testid="stSidebar"] [data-testid^="stCheckbox"],
    section[data-testid="stSidebar"] [data-testid^="stToggle"] {
        transition: filter 180ms ease, transform 180ms ease;
    }
    section[data-testid="stSidebar"] [data-testid^="stCheckbox"]:hover,
    section[data-testid="stSidebar"] [data-testid^="stToggle"]:hover {
        filter: drop-shadow(0 0 6px rgba(106,169,255,0.18));
        transform: translateY(-1px);
    }

    /* Reset to default radios (no custom pill styling) */
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Simple Login Gate --------------------
# Demo credentials (change for production)
APP_USERNAME = "demo"
APP_PASSWORD = "Pass@123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "users" not in st.session_state:
    st.session_state.users = {APP_USERNAME: APP_PASSWORD}
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

def render_login():
    st.markdown(
        """
        <style>
        .login-wrap {
            /* Place the form just below the title, centered horizontally */
            min-height: unset;
            padding-top: 12px;
            display: flex; align-items: flex-start; justify-content: center;
        }
        .login-card {
            width: min(420px, 92vw);
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 22px 20px 18px 20px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04);
            backdrop-filter: blur(6px);
        }
        .login-title { text-align:center; margin: 6px 0 4px 0; }
        .login-title span {
            background: linear-gradient(90deg, #93c5fd 0%, #7dd3fc 55%, #a5b4fc 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 900; font-size: clamp(26px, 4.6vw, 34px);
            letter-spacing: 0.3px;
        }
        .login-sub { text-align:center; opacity: 0.9; margin-bottom: 6px; }
        /* Inputs: full width + uniform height */
        .stTextInput > div > div { width: 100% !important; }
        .stTextInput input { height: 46px; font-size: 16px; width: 100% !important; }
        /* Submit button: full width + uniform height */
        [data-testid="stFormSubmitButton"] button, .login-btn button {
            width: 100% !important; height: 46px; font-size: 16px;
        }
        /* Hard-center Streamlit form container and remove default wide frame */
        [data-testid="stForm"] {
            max-width: 520px;
            margin-left: auto; margin-right: auto;
            background: transparent !important;
            border: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stForm"] > div { padding: 0 !important; }
        /* Compress vertical gaps between inputs and button */
        [data-testid="stForm"] label { margin-bottom: 4px !important; }
        [data-testid="stForm"] .stTextInput { margin-bottom: 8px !important; }
        [data-testid="stForm"] button { margin-top: 8px !important; }
        /* Center the Create Account CTA and signup area */
        .login-cta { display:flex; justify-content:center; margin-top: 8px; }
        .login-cta .stButton button { width: auto !important; padding: 8px 14px; }
        .signup-wrap { margin-top: 8px; }
        .signup-wrap h4 { text-align:center; margin: 6px 0 6px 0; }
        </style>
        <div class="login-wrap">
          <div class="login-card">
            <div style="display:flex; align-items:center; justify-content:center; gap:10px;">
              <div class='brand-icon'><span>🎵</span></div>
              <div class="login-title"><span>AI Music Composer</span></div>
            </div>
            <div class="login-sub">Sign in to continue</div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            if username in st.session_state.users and st.session_state.users.get(username) == password:
                st.session_state.authenticated = True
                st.success("Signed in successfully. Redirecting…")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Signup toggle (centered using columns)
    _cta_left, _cta_mid, _cta_right = st.columns([1, 2, 1])
    with _cta_mid:
        if st.button("Create account", key="create_account_btn", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()

    if st.session_state.show_signup:
        _su_left, _su_mid, _su_right = st.columns([1, 2, 1])
        with _su_mid:
            st.markdown("<div class='signup-wrap'>", unsafe_allow_html=True)
            st.markdown("<h4>Create your account</h4>", unsafe_allow_html=True)
            with st.form("signup_form", clear_on_submit=True):
                new_user = st.text_input("Choose a username", placeholder="Create username")
                new_pass = st.text_input("Choose a password", type="password", placeholder="Create password")
                new_pass2 = st.text_input("Confirm password", type="password", placeholder="Confirm password")
                create_clicked = st.form_submit_button("Create and Sign In", use_container_width=True)
                if create_clicked:
                    if not new_user or not new_pass:
                        st.warning("Please enter a username and password.")
                    elif new_pass != new_pass2:
                        st.warning("Passwords do not match.")
                    elif new_user in st.session_state.users:
                        st.warning("Username already exists. Choose another.")
                    else:
                        st.session_state.users[new_user] = new_pass
                        st.session_state.authenticated = True
                        st.session_state.show_signup = False
                        st.success("Account created. Redirecting…")
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

if not st.session_state.authenticated:
    render_login()
    st.stop()

# -------------------- Load Artifacts (cached) --------------------
MODEL_PATHS = [
    "models/lstm_note_model.keras",  # preferred
    "lstm_note_model.keras",         # fallback
]
MAPPING_PATH = "note_mappings.pkl"

@st.cache_resource(show_spinner=False)
def load_model_cached():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return tf.keras.models.load_model(p), p
    return None, None

@st.cache_data(show_spinner=False)
def load_mappings_cached():
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "rb") as f:
            unique_tokens, token_to_int = pickle.load(f)
        int_to_token = {i: t for t, i in token_to_int.items()}
        return unique_tokens, token_to_int, int_to_token
    return [], {}, {}

model, model_path = load_model_cached()
unique_tokens, token_to_int, int_to_token = load_mappings_cached()

# Try to infer sequence length from the model
DEFAULT_SEQ_LEN = 50

def infer_sequence_length(m):
    try:
        if hasattr(m, "input_shape") and m.input_shape is not None:
            # input_shape like (None, seq_len)
            if isinstance(m.input_shape, (list, tuple)) and len(m.input_shape) > 1:
                return int(m.input_shape[1])
        # Fallback: first layer input_length (Embedding)
        if len(m.layers) and hasattr(m.layers[0], "input_length") and m.layers[0].input_length:
            return int(m.layers[0].input_length)
    except Exception:
        pass
    return DEFAULT_SEQ_LEN

SEQ_LEN_INFERRED = infer_sequence_length(model) if model is not None else DEFAULT_SEQ_LEN

# -------------------- Sidebar Controls --------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>ACCOUNT</div>", unsafe_allow_html=True)
    st.success(f"Logged in as {APP_USERNAME}")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

st.sidebar.markdown("<div class='sidebar-title'>CONTROLS</div>", unsafe_allow_html=True)
GENRES = ["Classical", "Jazz", "Rock", "EDM"]
genre = st.sidebar.radio("SELECT THE OPTIONS", GENRES, index=0)
TARGET_BPM = 120  # fixed tempo used for timing conversion
SECONDS_MIN, SECONDS_MAX, SECONDS_DEFAULT = 5, 120, 30
duration_seconds = st.sidebar.slider("Time (in seconds)", min_value=SECONDS_MIN, max_value=SECONDS_MAX, value=SECONDS_DEFAULT, step=1)
DEFAULT_TEMPERATURE = 0.9
SEED_MODE_DEFAULT = "Random window"
temperature = DEFAULT_TEMPERATURE
sequence_length = SEQ_LEN_INFERRED
seed_mode = SEED_MODE_DEFAULT

st.sidebar.markdown("---")
show_progress = st.sidebar.toggle("Show generation progress", value=True)
render_audio = st.sidebar.toggle("Render WAV preview (if available)", value=True and HAS_AUDIO)

# -------------------- Header --------------------
st.markdown("""
<div class='brand-wrap'>
  <div class='brand-header'>
    <div class='brand-icon'><span>🎵</span></div>
    <h1 class='brand-title'>AI Music Composer</h1>
  </div>
  <div class='brand-subtitle'>Neural sequence model composing symbolic music tokens, rendered to MIDI/WAV.</div>
</div>
""", unsafe_allow_html=True)

# Centered metrics row under title
st.markdown(
    f"""
    <div style="display:flex; justify-content:center; gap:18px; margin: 10px 0 14px 0; flex-wrap:wrap;">
      <div class='metric-card' style='min-width:210px; text-align:center;'>
        <div style='font-size:13px; opacity:0.85;'>Model</div>
        <div style='font-size:17px; font-weight:700;'>{os.path.basename(model_path) if model_path else 'Not loaded'}</div>
      </div>
      <div class='metric-card' style='min-width:210px; text-align:center;'>
        <div style='font-size:13px; opacity:0.85;'>Vocab size</div>
        <div style='font-size:17px; font-weight:700;'>{len(unique_tokens) if unique_tokens else 0}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Generation Utilities --------------------
def sample_from_probs(probs, temperature_value=1.0):
    probs = np.asarray(probs).astype("float64")
    if temperature_value <= 0:
        return int(np.argmax(probs))
    logits = np.log(probs + 1e-12) / float(temperature_value)
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    return int(np.random.choice(len(probs), p=probs))

def build_seed(sequence_len):
    if not unique_tokens or not token_to_int:
        return []
    if seed_mode == "Single token (repeat)":
        idx = np.random.randint(0, len(unique_tokens))
        token_idx = token_to_int[unique_tokens[idx]]
        return [token_idx] * sequence_len
    # Random window of tokens (independent)
    return [token_to_int[unique_tokens[np.random.randint(0, len(unique_tokens))]] for _ in range(sequence_len)]

def generate_tokens(model_obj, sequence_len, total_length, temp):
    if not unique_tokens:
        return []
    pattern = build_seed(sequence_len)
    if not pattern:
        return []
    output_tokens = []
    prog = st.progress(0.0) if show_progress else None
    for i in range(total_length):
        x = np.array([pattern[-sequence_len:]], dtype=np.int32)
        preds = model_obj.predict(x, verbose=0)[0]
        next_idx = sample_from_probs(preds, temperature_value=temp)
        output_tokens.append(int_to_token.get(next_idx, unique_tokens[0]))
        pattern.append(next_idx)
        if prog is not None:
            prog.progress((i + 1) / total_length)
    if prog is not None:
        prog.empty()
    return output_tokens

def tokens_to_midi(prediction_output, output_path="outputs/music.mid", bpm=120, duration_seconds=None, quarters_per_token=1.0):
    midi_stream = m21_stream.Stream()
    # Set a fixed tempo for accurate duration in seconds
    midi_stream.append(m21_tempo.MetronomeMark(number=bpm))
    # If a target duration is provided, distribute evenly across tokens
    total_quarters = None
    if duration_seconds is not None and len(prediction_output) > 0:
        total_quarters = (duration_seconds * bpm) / 60.0
        base_q_per_token = max(total_quarters / float(len(prediction_output)), 1e-6)
    else:
        base_q_per_token = quarters_per_token
    current_offset = 0.0
    for idx, t in enumerate(prediction_output):
        try:
            if total_quarters is not None and idx == len(prediction_output) - 1:
                # Adjust last token so the sum of durations is exact
                used_quarters = base_q_per_token * (len(prediction_output) - 1)
                q_len = max(total_quarters - used_quarters, 1e-6)
            else:
                q_len = base_q_per_token
            if ("." in t) or t.isdigit():
                # Treat token as chord of pitch classes or digits exactly as encoded
                parts = t.split(".")
                ch = chord.Chord(parts)
                ch.storedInstrument = instrument.Piano()
                ch.quarterLength = q_len
                ch.offset = current_offset
                midi_stream.append(ch)
            else:
                n = note.Note(t)
                n.storedInstrument = instrument.Piano()
                n.quarterLength = q_len
                n.offset = current_offset
                midi_stream.append(n)
            current_offset += q_len
        except Exception:
            continue
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi_stream.write("midi", fp=output_path)
    return output_path

def midi_to_wav(midi_path, wav_path="outputs/wav/generated_music.wav", fs=44100, duration_seconds=None):
    if not HAS_AUDIO:
        return None
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    midi_obj = pm.PrettyMIDI(midi_path)
    audio = midi_obj.synthesize(fs=fs)
    if duration_seconds is not None and duration_seconds > 0:
        target_samples = int(round(duration_seconds * fs))
        if len(audio) < target_samples:
            pad = target_samples - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant', constant_values=0.0)
        elif len(audio) > target_samples:
            audio = audio[:target_samples]
    sf.write(wav_path, audio, fs)
    return wav_path

# -------------------- Action --------------------
# Derive token length from desired seconds given fixed BPM and 1 token per beat
tokens_per_second = TARGET_BPM / 60.0  # 1 token per quarter note (beat)
length = int(np.ceil(duration_seconds * tokens_per_second))

_c1, _c2, _c3 = st.columns([0.35, 0.30, 0.35])
with _c2:
    generate_clicked = st.button("✨ Compose Music", use_container_width=True)
    st.markdown(
        f"<div style='text-align:center; margin-top: 10px; font-size: 16px; white-space: nowrap;'>Time: {duration_seconds}s  (≈ {length} tokens at {TARGET_BPM} BPM)</div>",
        unsafe_allow_html=True,
    )

if generate_clicked:
    if model is None:
        st.error("❌ No trained model found! Train the model first.")
    elif not unique_tokens:
        st.error("❌ No token mappings found! Run training notebook and export mappings.")
    else:
        with st.spinner(f"Composing {genre} style music..."):
            generated_tokens = generate_tokens(model, sequence_length, length, temperature)
            output_mid = f"outputs/{genre}_music.mid"
            midi_file = tokens_to_midi(
                generated_tokens,
                output_path=output_mid,
                bpm=TARGET_BPM,
                duration_seconds=duration_seconds,
                quarters_per_token=1.0,
            )

        st.success(f"✅ Music saved as {output_mid}")

        # Downloads
        with open(output_mid, "rb") as f:
            st.markdown("<div class='download-wrap'>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download MIDI",
                data=f,
                file_name=f"{genre}_music.mid",
                mime="audio/midi",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Optional WAV render and audio playback
        if render_audio and HAS_AUDIO:
            with st.spinner("Rendering audio preview (WAV)..."):
                wav_path = midi_to_wav(
                    midi_file,
                    wav_path=f"outputs/wav/{genre}_music.wav",
                    duration_seconds=duration_seconds,
                )
            if wav_path and os.path.exists(wav_path):
                st.audio(wav_path)
                with open(wav_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download WAV",
                        data=f,
                        file_name=f"{genre}_music.wav",
                        mime="audio/wav",
                    )

# -------------------- Footer / Tips --------------------
st.markdown("""
<div style='margin-top: 28px; opacity: 0.9; display:flex; justify-content:center;'>
  <div style='max-width: 1000px; width: 100%; text-align: center; font-size: 16px;'>
    <span class='pill'>Info</span>
    The composer uses tuned sampling settings for a balance of creativity and coherence.
  </div>
</div>
""", unsafe_allow_html=True)

