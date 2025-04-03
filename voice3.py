import os
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

# Configuration
PHRASE = "I let the positive overrun the negative"
SAMPLE_RATE = 16000
DURATION = 5 
TRAIN_SAMPLES = 4
VOICE_FOLDER = "user"  # Folder to store voice biometric data
VOICE_MODEL_FILE = os.path.join(VOICE_FOLDER, "voice_model.pkl")  # Fixed file name for voice model

# Ensure the voice folder exists
os.makedirs(VOICE_FOLDER, exist_ok=True)

# Terminal-based audio recording function
def record_audio(output_file, duration=DURATION, fs=SAMPLE_RATE):
    """Record audio from terminal and save to file"""
    print(f"\nPlease say: \"{PHRASE}\"")
    print("Recording will start in...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording NOW! 🎤")
    
    # Record audio
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    
    # Show progress bar
    for i in range(duration):
        time.sleep(1)
        bars = "█" * (i + 1) + "░" * (duration - i - 1)
        print(f"\rRecording: {bars} {i+1}/{duration}s", end="", flush=True)
    
    # Wait for recording to complete
    sd.wait()
    print("\n✅ Recording complete!")
    
    # Save audio to file
    sf.write(output_file, audio_data, fs)
    print(f"✅ Saved to {output_file}")
    
    return output_file

# Extract enhanced features
def extract_enhanced_features(file_path, n_mfcc=20):
    """Extract acoustic features from audio file"""
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Add delta and delta-delta features (velocity and acceleration)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Combine features into a single vector
    combined_features = np.vstack([
        np.mean(mfcc.T, axis=0),      # Mean of MFCCs
        np.std(mfcc.T, axis=0),       # Standard deviation of MFCCs
        np.mean(delta_mfcc.T, axis=0),  # Mean of deltas
        np.mean(delta2_mfcc.T, axis=0)  # Mean of delta-deltas
    ])
    
    return combined_features.flatten()

# Register voice biometrics function
def register_voice_biometrics(train_dir=os.path.join(VOICE_FOLDER, "train")):
    """Record voice samples and create a voice biometric model"""
    print("\n=== VOICE BIOMETRICS REGISTRATION ===")
    print(f"You will record the phrase \"{PHRASE}\" {TRAIN_SAMPLES} times")
    
    # Ensure directory exists
    os.makedirs(train_dir, exist_ok=True)
    
    features = []
    
    # Record training samples
    for i in range(1, TRAIN_SAMPLES + 1):
        print(f"\nTraining sample {i}/{TRAIN_SAMPLES}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(train_dir, f"train_{i}_{timestamp}.wav")
        
        try:
            record_audio(output_file)
            
            # Extract features
            user_features = extract_enhanced_features(output_file)
            features.append(user_features)
            print(f"Processed: {os.path.basename(output_file)}")
        except Exception as e:
            print(f"Error processing recording {i}: {str(e)}")
            print("Please try again.")
            i -= 1  # Retry this recording
            continue
    
    if len(features) < 2:
        print("❌ Not enough valid recordings to create a voice profile")
        return False
        
    features = np.array(features)
    print(f"\nTraining with {len(features)} audio samples, feature dimension: {features.shape}")
    
    # Display a loading indicator
    print("Training model... ", end="", flush=True)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create GMM with proper regularization
    gmm = GaussianMixture(
        n_components=min(2, len(features)-1),  # Adaptive number of components
        covariance_type='diag',   # Diagonal is more stable than full
        n_init=10,
        reg_covar=1.0,
        random_state=42,
        max_iter=300
    )
    
    try:
        gmm.fit(features_scaled)
        
        # Save both the model and scaler together in the fixed location
        with open(VOICE_MODEL_FILE, "wb") as f:
            pickle.dump((gmm, scaler, PHRASE), f)
        print("DONE!")
        print(f"✅ Voice biometric data saved successfully to {VOICE_MODEL_FILE}")
        return True
    except Exception as e:
        print("FAILED!")
        print(f"❌ Error training model: {str(e)}")
        return False

# Check if voice biometrics exist
def has_voice_biometrics():
    """Check if voice biometric data already exists"""
    return os.path.exists(VOICE_MODEL_FILE)

# Main Execution - Registration Only
if __name__ == "__main__":
    print("=== Voice Biometrics Registration ===")
    
    if has_voice_biometrics():
        print("Voice biometric data already exists.")
        choice = input("Do you want to re-register your voice? (y/n): ")
        if choice.lower() != 'y':
            print("Registration cancelled. Existing voice data retained.")
            exit()
        print("Proceeding with re-registration...")
    else:
        print("No voice biometric data found. Creating new registration.")
    
    # Proceed with registration
    success = register_voice_biometrics()
    
    if success:
        print("\nVoice registration completed successfully.")
        print(f"Model saved to {VOICE_MODEL_FILE}")
    else:
        print("\nVoice registration failed. Please try again.")