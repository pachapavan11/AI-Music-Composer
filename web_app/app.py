from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import secrets
import os
import threading
from music_generator import load_models_data, generate_tokens, tokens_to_midi, midi_to_wav

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
GENRES = ['Classical', 'Jazz', 'Pop', 'Rock']

# Model Paths
GENRE_TO_MODEL = {
    "Classical": "models/lstm_classical_best.keras",
    "Jazz": "models/lstm_jazz_best.keras",
    "Rock": "models/lstm_rock_best.keras",
    "Pop": "models/lstm_pop_best.keras",
}

GENRE_TO_MAPPING_PKL = {
    "Classical": "models/classical_note_mappings.pkl",
    "Jazz": "models/jazz_note_mappings.pkl",
    "Rock": "models/rock_note_mappings.pkl",
    "Pop": "models/pop_note_mappings.pkl",
}

GENRE_CONFIG = {
    "Classical": {"temp": 0.9, "note_len": 1.0, "bpm": 100},
    "Jazz": {"temp": 1.1, "note_len": 0.8, "bpm": 110},
    "Rock": {"temp": 1.2, "note_len": 0.6, "bpm": 130},
    "Pop": {"temp": 1.0, "note_len": 0.7, "bpm": 125},
}

# Output directories
OUTPUT_DIR = "web_app/static/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)
WAV_DIR = "web_app/static/generated/wav"
os.makedirs(WAV_DIR, exist_ok=True)

# Load models at startup
print("Loading models...")
MODELS_CACHE = load_models_data(GENRE_TO_MODEL, GENRE_TO_MAPPING_PKL)
print("Models loaded.")

@app.route('/')
def index():
    """Login page with captcha."""
    num1 = secrets.randbelow(10)
    num2 = secrets.randbelow(10)
    session['captcha_result'] = num1 + num2
    return render_template('index.html', captcha_question=f"{num1} + {num2}")

@app.route('/login', methods=['POST'])
def login():
    """Handle login and captcha verification."""
    username = request.form.get('username')
    captcha_answer = request.form.get('captcha')
    
    if not username or not captcha_answer:
        return jsonify({'success': False, 'message': 'Please fill all fields'})
    
    try:
        if int(captcha_answer) != session.get('captcha_result'):
            return jsonify({'success': False, 'message': 'Incorrect captcha'})
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid captcha format'})
        
    session['username'] = username
    return jsonify({'success': True, 'redirect': url_for('composer')})

@app.route('/composer')
def composer():
    """Music composer interface."""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('composer.html', genres=GENRES, username=session['username'])

@app.route('/logout')
def logout():
    """Logout user."""
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/api/compose', methods=['POST'])
def compose_api():
    """API endpoint for music composition."""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    data = request.json
    genre = data.get('genre')
    duration = data.get('duration')
    
    # Validation
    if not genre or genre not in GENRES:
        return jsonify({'success': False, 'message': 'Invalid genre'})
        
    try:
        duration = int(duration)
        if duration < 5 or duration > 90:
             return jsonify({'success': False, 'message': 'Duration must be between 5 and 90 seconds'})
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Invalid duration'})

    # Define output filenames
    base_name = f"{genre}_{secrets.token_hex(4)}"
    midi_filename = f"{base_name}.mid"
    wav_filename = f"{base_name}.wav"
    
    try:
        if genre not in MODELS_CACHE:
             return jsonify({'success': False, 'message': f'Model for {genre} not loaded.'})

        wrapper = MODELS_CACHE[genre]
        config = GENRE_CONFIG[genre]
        
        # Calculate target beats
        tps = config["bpm"] / 60 
        target_beats = duration * tps
        
        # Generate tokens
        tokens = generate_tokens(
            wrapper,
            seq_len=wrapper.model.input_shape[1],
            max_duration_beats=target_beats,
            temp=config['temp']
        )
        
        # Save MIDI
        midi_path = os.path.join(OUTPUT_DIR, midi_filename)
        tokens_to_midi(tokens, midi_path, config['bpm'], config['note_len'])
        
        # Download URLs
        midi_url = url_for('static', filename=f'generated/{midi_filename}')
        
        response_msg = f'Music composed! <a href="{midi_url}" target="_blank">Download MIDI</a>'
        
        # Try WAV
        wav_path = os.path.join(WAV_DIR, wav_filename)
        generated_wav = midi_to_wav(midi_path, wav_path)
        
        response_data = {
            'success': True,
            'message': 'Music generated successfully!',
            'midi_url': midi_url,
            'wav_url': None
        }

        if generated_wav:
             wav_url = url_for('static', filename=f'generated/wav/{wav_filename}')
             response_data['wav_url'] = wav_url
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Generation failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
