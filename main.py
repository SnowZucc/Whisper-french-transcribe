# Inspiré de https://huggingface.co/bofenghuang/whisper-medium-french

import torch
from datasets import load_dataset
from transformers import pipeline
import librosa  # Import librosa
import soundfile as sf  # Import soundfile

# Initialize the pipeline globally to keep it in memory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading pipeline...")
global_pipe = pipeline("automatic-speech-recognition", model="bofenghuang/whisper-medium-french", device=device)
global_pipe.model.config.forced_decoder_ids = global_pipe.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")
print("Pipeline loaded and configured successfully.")

def transcribe_audio(mp3_file):
    print(f"Loading MP3 file: {mp3_file}")
    try:
        waveform, sample_rate = librosa.load(mp3_file, sr=None)  # Load the MP3 file using librosa
        print(f"MP3 file loaded. Sample rate: {sample_rate}, Waveform shape: {waveform.shape}")
    except Exception as e:
        print(f"Error loading MP3 file: {e}")
        return None

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
        print("Resampling complete.")

    # Convert to mono if necessary
    if len(waveform.shape) > 1:
        print("Converting to mono...")
        waveform = librosa.to_mono(waveform)
        print("Conversion to mono complete.")

    # Prepare the audio data as a dictionary
    audio_data = {"array": waveform, "sampling_rate": sample_rate}
    print(f"Audio data prepared. Sampling rate: {audio_data['sampling_rate']}, Array length: {len(audio_data['array'])}")

    # Run transcription
    print("Running transcription pipeline...")
    try:
        generated_sentences = global_pipe(audio_data, max_new_tokens=225)["text"]  # greedy
        print("Transcription complete.")
        return generated_sentences
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Example usage
mp3_file = ""  # Replace with the actual path to your MP3 file
generated_sentences = transcribe_audio(mp3_file)

if generated_sentences:
    print("Generated sentences:")
    print(generated_sentences)