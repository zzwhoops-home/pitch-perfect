import time
import numpy as np
import sounddevice as sd
import aubio

# --- Configuration ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
HOP_SIZE = 512

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 2-Octave Major Scale Intervals (15 notes up + 14 notes down = 29 total steps)
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24]
FULL_SCALE_STEPS = MAJOR_SCALE_INTERVALS + MAJOR_SCALE_INTERVALS[-2::-1]

pitch_detector = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_silence(-40)

def hz_to_note_and_cents(freq: float):
    if freq <= 0:
        return None, None, None, None
    midi_float = 69 + 12 * np.log2(freq / 440.0)
    midi_note = int(round(midi_float))
    target_freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    cents = 1200 * np.log2(freq / target_freq)
    note_name = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{note_name}{octave}", cents, target_freq, midi_note

def midi_to_note_name(midi_note: int) -> str:
    return f"{NOTE_NAMES[midi_note % 12]}{(midi_note // 12) - 1}"

# --- Scale Tracking Variables ---
state = "WAITING_FOR_ROOT"
root_midi = None
current_step_idx = 0

# Sample collection for current note
current_note_samples = []
last_note_capture_time = 0

results = []

def audio_callback(indata, frames, time_info, status):
    global state, root_midi, current_step_idx, current_note_samples, last_note_capture_time
    
    signal = indata[:, 0].astype(np.float32)
    pitch = pitch_detector(signal)[0]
    confidence = pitch_detector.get_confidence()

    # Match threshold from your working code
    if pitch > 50 and confidence > 0.6:
        note_str, cents, _, midi_note = hz_to_note_and_cents(pitch)
        now = time.time()

        # STATE 1: Waiting to register any note played to set the scale root
        if state == "WAITING_FOR_ROOT":
            root_midi = midi_note
            state = "RECORDING"
            print(f"\nROOT DETECTED: {midi_to_note_name(root_midi)}")
            print(f"Play Step 1/29: {midi_to_note_name(root_midi + FULL_SCALE_STEPS[0])}\n")
            current_note_samples = []
            last_note_capture_time = now

        # STATE 2: Scale progression tracking
        elif state == "RECORDING":
            target_midi = root_midi + FULL_SCALE_STEPS[current_step_idx]
            target_name = midi_to_note_name(target_midi)

            # Live feedback print
            status_str = "IN TUNE" if abs(cents) <= 10 else ("SHARP" if cents > 0 else "FLAT")
            print(f"\rTarget: {target_name:<4} | Heard: {note_str:<4} ({pitch:5.1f} Hz) | Dev: {cents:+5.1f}c | [{status_str:<7}]", end="")

            # If heard pitch is close to expected target note (within ±1 semitone)
            if abs(midi_note - target_midi) <= 1:
                current_note_samples.append(cents)

                # Capture note once we have 5 consistent frame readings (~0.05 seconds) and 0.3s cooldown
                if len(current_note_samples) >= 5 and (now - last_note_capture_time) > 0.3:
                    avg_cents = float(np.mean(current_note_samples))
                    
                    print(f"\n  ✓ Step {current_step_idx + 1:02d}/29 recorded: {target_name} ({avg_cents:+5.1f} cents)")
                    
                    results.append({
                        "step": current_step_idx + 1,
                        "target": target_name,
                        "cents": avg_cents
                    })
                    
                    current_step_idx += 1
                    current_note_samples = []
                    last_note_capture_time = now

                    if current_step_idx >= len(FULL_SCALE_STEPS):
                        state = "FINISHED"
                    else:
                        next_target = midi_to_note_name(root_midi + FULL_SCALE_STEPS[current_step_idx])
                        print(f"👉 Next Note: {next_target}")

# Run Stream
print("Listening... Play ANY starting note (e.g. G3 or A3) to begin your 2-octave scale!")
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=HOP_SIZE, callback=audio_callback):
    try:
        while state != "FINISHED":
            sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped early.")

# --- Final Grade Report ---
if results:
    print("\n" + "="*45)
    print("       VIOLIN SCALE INTONATION REPORT       ")
    print("="*45)
    
    avg_error = np.mean([abs(r["cents"]) for r in results])
    score = max(0.0, 100.0 - (avg_error * 2.0))
    
    print(f"Notes Logged     : {len(results)} / 29")
    print(f"Average Error    : {avg_error:.1f} cents")
    print(f"Overall Accuracy : {score:.1f}%\n")
    
    # Show worst and best notes
    results_sorted = sorted(results, key=lambda x: abs(x["cents"]))
    print(f"Best Intonation  : {results_sorted[0]['target']} ({results_sorted[0]['cents']:+.1f}c)")
    print(f"Worst Intonation : {results_sorted[-1]['target']} ({results_sorted[-1]['cents']:+.1f}c)")
    print("="*45)