document.addEventListener('DOMContentLoaded', () => {
    initBackground();
    setupForms();
});

function initBackground() {
    const container = document.getElementById('background-notes');
    if (!container) return;

    const symbols = ['♪', '♫', '♬', '♩', '♭', '♮', '♯', '𝄞', '𝄢'];

    // Create initial batch
    for (let i = 0; i < 20; i++) {
        createNote(container, symbols);
    }

    // Continuously create new notes
    setInterval(() => {
        createNote(container, symbols);
    }, 2000);
}

function createNote(container, symbols) {
    const note = document.createElement('div');
    note.classList.add('note');
    note.innerText = symbols[Math.floor(Math.random() * symbols.length)];

    // Random positioning
    note.style.left = Math.random() * 100 + 'vw';
    note.style.animationDuration = (Math.random() * 5 + 10) + 's'; // Slower float
    note.style.fontSize = (Math.random() * 2 + 1) + 'rem';

    container.appendChild(note);

    // Cleanup
    setTimeout(() => {
        note.remove();
    }, 15000);
}

function setupForms() {
    const loginForm = document.getElementById('login-form');
    const composerForm = document.getElementById('composer-form');

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(loginForm);
            const errorMsg = document.getElementById('error-msg');
            const submitBtn = loginForm.querySelector('button[type="submit"]');

            errorMsg.innerText = '';
            submitBtn.disabled = true;
            submitBtn.innerText = 'Verifying...';

            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (result.success) {
                    window.location.href = result.redirect;
                } else {
                    errorMsg.innerText = result.message;
                    submitBtn.disabled = false;
                    submitBtn.innerText = 'Enter Studio';
                }
            } catch (err) {
                errorMsg.innerText = 'Server connection error';
                submitBtn.disabled = false;
                submitBtn.innerText = 'Enter Studio';
            }
        });
    }

    if (composerForm) {
        composerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('compose-btn');
            const statusText = document.getElementById('status-text');
            const statusIcon = document.getElementById('status-icon');
            const visualizer = document.getElementById('visualizer');
            const audioContainer = document.getElementById('audio-player-container');
            const audioPlayer = document.getElementById('audio-player');
            const downloadLinks = document.getElementById('download-links');

            // UI Loading State
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Composing...';

            visualizer.style.display = 'flex';
            statusIcon.style.display = 'none';
            audioContainer.style.display = 'none';
            statusText.innerHTML = '<h3>Summoning the Muses...</h3><p>Analyzing harmonic structures and generating notes.</p>';

            const data = {
                genre: document.getElementById('genre').value,
                duration: document.getElementById('duration').value
            };

            try {
                const response = await fetch('/api/compose', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                if (result.success) {
                    statusText.innerHTML = `<h3>Masterpiece Created!</h3><p>${result.message}</p><p style="font-size: 0.85rem; color: #dfe6e9; margin-top: 10px; font-style: italic;">Note: Mentioned seconds and generated seconds has difference due to note density</p>`;
                    btn.innerHTML = '<i class="fas fa-magic"></i> Compose Another';

                    // Show Audio Player
                    audioContainer.style.display = 'block';

                    if (result.wav_url) {
                        audioPlayer.src = result.wav_url;
                        audioPlayer.load();
                    } else {
                        statusText.innerHTML += '<p><small>(Audio preview unavailable, MIDI only)</small></p>';
                    }

                    // Update Links
                    let linksHtml = `<a href="${result.midi_url}" target="_blank"><i class="fas fa-file-audio"></i> Download MIDI</a>`;
                    if (result.wav_url) {
                        linksHtml += `<a href="${result.wav_url}" target="_blank"><i class="fas fa-music"></i> Download WAV</a>`;
                    }
                    downloadLinks.innerHTML = linksHtml;

                } else {
                    statusText.innerHTML = `<h3>Composition Failed</h3><p>${result.message}</p>`;
                    statusIcon.style.display = 'block';
                    statusIcon.innerHTML = '<i class="fas fa-exclamation-circle" style="color: #ff7675;"></i>';
                    btn.innerHTML = '<i class="fas fa-redo"></i> Try Again';
                }
            } catch (err) {
                statusText.innerHTML = '<h3>Connection Error</h3><p>Could not reach the conductor.</p>';
                statusIcon.style.display = 'block';
                statusIcon.innerHTML = '<i class="fas fa-wifi" style="color: #ff7675;"></i>';
                btn.innerHTML = '<i class="fas fa-redo"></i> Try Again';
                console.error(err);
            } finally {
                btn.disabled = false;
                visualizer.style.display = 'none';
                if (!audioContainer.style.display || audioContainer.style.display === 'none') {
                    statusIcon.style.display = 'block';
                }
            }
        });

        // Genre Change Effects
        const genreSelect = document.getElementById('genre');
        if (genreSelect) {
            genreSelect.addEventListener('change', (e) => {
                const genre = e.target.value;
                const statusText = document.getElementById('status-text');

                // Reset animation
                const notes = document.querySelectorAll('.note');
                notes.forEach(n => {
                    n.style.animation = 'none';
                    n.offsetHeight; /* trigger reflow */
                    n.style.animation = null;
                });

                let themeColor = '';
                let message = '';

                switch (genre) {
                    case 'Classical':
                        themeColor = '#fdcb6e'; // Gold/Yellow
                        message = 'Refined elegance.';
                        break;
                    case 'Jazz':
                        themeColor = '#e17055'; // Burnt Orange
                        message = 'Smooth and improvised.';
                        break;
                    case 'Rock':
                        themeColor = '#d63031'; // Red
                        message = 'Energetic and bold.';
                        break;
                    case 'Pop':
                        themeColor = '#0984e3'; // Blue
                        message = 'Catchy and upbeat.';
                        break;
                }

                // Update CSS variable slightly for effect
                document.documentElement.style.setProperty('--secondary-color', themeColor);

                // Update text with fade
                if (statusText) {
                    statusText.style.opacity = 0;
                    setTimeout(() => {
                        statusText.innerHTML = `<h3>${genre} Mode</h3><p>${message}</p>`;
                        statusText.style.opacity = 1;
                    }, 300);
                }
            });
        }
    }
}
