import numpy as np
import sounddevice as sd
import aubio

# Configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048  # Frame size for pitch detection
HOP_SIZE = 512      # Shift size

# Initialize Aubio pitch tracker (YIN algorithm works great for strings)
pitch_detector = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_silence(-40)  # Silence threshold in dB

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def hz_to_note_and_cents(freq: float):
    """Calculates nearest MIDI note, note name, target frequency, and cent deviation."""
    if freq <= 0:
        return None, None, None
    
    # Calculate continuous MIDI pitch
    midi_float = 69 + 12 * np.log2(freq / 440.0)
    midi_note = int(round(midi_float))
    
    # Target frequency of exact equal temperament note
    target_freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    # Cents deviation: positive = sharp, negative = flat
    cents = 1200 * np.log2(freq / target_freq)
    
    note_name = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    full_note = f"{note_name}{octave}"
    
    return full_note, cents, target_freq

def audio_callback(indata, frames, time, status):
    """Callback function processing audio chunks in real time."""
    signal = indata[:, 0].astype(np.float32)
    
    # Extract pitch
    pitch = pitch_detector(signal)[0]
    confidence = pitch_detector.get_confidence()
    
    # Filter out weak signals/silence
    if pitch > 50 and confidence > 0.7:  # Violin open G string is ~196 Hz (G3)
        note, cents, _ = hz_to_note_and_cents(pitch)
        
        # Simple feedback output
        status_str = "IN TUNE" if abs(cents) <= 10 else ("SHARP" if cents > 0 else "FLAT")
        print(f"Detected: {note:<4} | Pitch: {pitch:6.1f} Hz | Deviation: {cents:+5.1f} cents | [{status_str}]")

# Start listening
print("Listening... Play your scale! (Press Ctrl+C to stop)")
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=HOP_SIZE, callback=audio_callback):
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")