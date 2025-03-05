import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
from sklearn.model_selection import train_test_split

# Set dataset directory
dataset_path = "LieWaves"

# Path to the raw EEG data
raw_data_path = os.path.join(dataset_path, "Raw")

# List all available EEG files
eeg_files = sorted([f for f in os.listdir(raw_data_path) if f.endswith(".csv")])

# Load an example file (Subject 1, Experiment 1)
sample_file = os.path.join(raw_data_path, eeg_files[0])  # Change index for different subjects
df = pd.read_csv(sample_file)

# Define channel names (Emotiv Insight has 5 channels)
channel_names = ['AF3', 'AF4', 'T7', 'T8', 'Pz']  

# Check if there's a time column (assuming 1st column is time)
if 'time' in df.columns[0].lower():
    df = df.iloc[:, 1:]  # Remove time column

# Transpose data to match MNE format (shape: channels x samples)
data = df.values.T  

# Sampling frequency (from dataset README)
sfreq = 128

mne.set_log_level("WARNING")

# Create MNE Info object
info = mne.create_info(channel_names, sfreq, ch_types='eeg')

# Convert to MNE RawArray
raw = mne.io.RawArray(data, info)

# Path to Subject_Stimuli.xlsx
stimuli_file = os.path.join(dataset_path, "Subject_Stimuli.xlsx")

# Read Excel sheet
stimuli_df = pd.read_excel(stimuli_file)

# Dictionary to store label information
file_label_mapping = {}

for _, row in stimuli_df.iterrows():
    subject_id = row['SUBJECT']  # Already in the correct format ("S1", "S2", ...)
    experiment_id = row['SESSION']  # Already in the correct format ("S1", "S2", ...)

    # Corrected filename format (no extra "S")
    filename = f"{subject_id}{experiment_id}.csv"

    # Extract truth/lie label
    condition = row['LIE/TRUTH']
    
    # Store mapping
    file_label_mapping[filename] = condition


# Apply a bandpass filter (0.5 - 45 Hz)
raw.filter(l_freq=0.5, h_freq=45, fir_design='firwin')

# Apply notch filter at 50 Hz and 60 Hz
raw.notch_filter(freqs=[50, 60], fir_design='firwin')

# Define parameters
window_length = 2  # Window length in seconds
overlap = 0.5  # 50% overlap
nperseg = int(window_length * sfreq)  # Number of samples per segment
noverlap = int(nperseg * overlap)  # Overlapping samples

# Compute Welch's PSD for each EEG channel
psds, freqs = psd_array_welch(
    raw.get_data(),   # EEG data (channels x time)
    sfreq=sfreq,      # Sampling frequency (128 Hz)
    fmin=0.5, fmax=45, # Frequency range of interest
    n_fft=nperseg,    # FFT segment length (2 seconds)
    n_overlap=noverlap, # Overlapping samples (50%)
    window='hamming', # Hamming window function
    average='mean'    # Average across windows
)


# Convert PSD results into a Pandas DataFrame
psd_df = pd.DataFrame(psds, index=channel_names, columns=freqs)

# Segment EEG data into overlapping windows
segmented_psd_data = []
segment_labels = []

# Iterate over EEG files
for filename, label in file_label_mapping.items():
    file_path = os.path.join(raw_data_path, filename)
    df = pd.read_csv(file_path)
    
    # Transpose to match MNE format
    data = df.values.T  

    # Sliding window segmentation
    for start in range(0, data.shape[1] - nperseg, noverlap):
        segment = data[:, start:start + nperseg]
        if segment.shape[1] == nperseg:  # Ensure full-length segments
            # Compute Welch's PSD for this segment
            psds, freqs = psd_array_welch(
                segment, 
                sfreq=sfreq, 
                fmin=0.5, fmax=45, 
                n_fft=nperseg, 
                n_overlap=noverlap, 
                window='hamming', 
                average='mean'
            )

            segmented_psd_data.append(psds)  # Store PSD features
            segment_labels.append(label)  # Store corresponding label

# Convert to NumPy arrays
segmented_psd_data = np.array(segmented_psd_data)  # Shape: (num_segments, channels, frequency_bins)
segment_labels = np.array(segment_labels)  # Shape: (num_segments,)

# Save PSD data instead of raw EEG segments
np.save(os.path.join(dataset_path, "segmented_psd.npy"), segmented_psd_data)
np.save(os.path.join(dataset_path, "segment_labels.npy"), segment_labels)

print("Segmented PSD data saved!")

# Load preprocessed PSD data
X = np.load(os.path.join(dataset_path, "segmented_psd.npy"))  # Shape: (num_segments, 5, num_frequencies)
y = np.load(os.path.join(dataset_path, "segment_labels.npy"))  # Shape: (num_segments,)

# Split into training (80%), validation (10%), test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Reshape data to include a channel dimension (needed for Conv2D)
X_train = X_train[..., np.newaxis]  # (3153, 5, 90, 1)
X_val = X_val[..., np.newaxis]      # (394, 5, 90, 1)
X_test = X_test[..., np.newaxis]    # (395, 5, 90, 1)

# Display sizes
print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Save split data
np.save(os.path.join(dataset_path, "X_train.npy"), X_train)
np.save(os.path.join(dataset_path, "y_train.npy"), y_train)
np.save(os.path.join(dataset_path, "X_val.npy"), X_val)
np.save(os.path.join(dataset_path, "y_val.npy"), y_val)
np.save(os.path.join(dataset_path, "X_test.npy"), X_test)
np.save(os.path.join(dataset_path, "y_test.npy"), y_test)


# Load the data
data = np.load("LieWaves/X_train.npy")

# Inspect the first 5 values
# print(data[:5])