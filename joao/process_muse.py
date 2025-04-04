import os
import numpy as np
import pandas as pd
import mne
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_welch


# ========== CONFIG ==========
sfreq = 256  # Muse sampling rate
window_length = 2  # seconds
nperseg = window_length * sfreq
noverlap = 0  # No overlap
group_size = 5  # For 5 consecutive PSDs
mne.set_log_level("WARNING")


# Add your CSV filenames here
true_files = ['joao_truth1.csv', 'joao_truth2.csv']
lie_files = ['antoine_lie1.csv', 'joao_lie1.csv']
eeg_channels = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
dataset_path = '.'  # Save output to current folder

# ========== FUNCTION ==========
def preprocess_muse_file(file_path):
    df = pd.read_csv(file_path)
    if 'Time' in df.columns[0]:
        df = df.iloc[:, 1:]
    data = df.values.T
    results = []

    # Apply filtering per channel
    for ch in data:
        ch = filter_data(ch, sfreq, l_freq=0.5, h_freq=45, verbose=False)
        ch = notch_filter(ch, Fs=sfreq, freqs=[50, 60], verbose=False)

        # Segment into non-overlapping windows
        for start in range(0, len(ch) - nperseg + 1, nperseg):
            segment = ch[start:start + nperseg]
            if len(segment) == nperseg and np.any(np.isfinite(segment)):

                psd, freqs = psd_array_welch(
                    segment[np.newaxis, :],
                    sfreq=sfreq,
                    fmin=0.5, fmax=45,
                    n_fft=nperseg,
                    n_overlap=noverlap,
                    window='hamming',
                    average='mean'
                )
                results.append(psd[0][:, np.newaxis])  # shape: (90, 1)
    return results

# ========== MAIN ==========
X_sequences = []
y_sequences = []

def build_sequences(psd_list, label):
    for i in range(0, len(psd_list) - group_size + 1, group_size):
        seq = psd_list[i:i + group_size]
        if len(seq) == group_size:
            X_sequences.append(np.stack(seq))  # shape: (5, 90, 1)
            y_sequences.append(label)

# Process all files
for file in true_files:
    psds = preprocess_muse_file(file)
    build_sequences(psds, label=1)

for file in lie_files:
    psds = preprocess_muse_file(file)
    build_sequences(psds, label=0)

# Convert and shuffle
X = np.array(X_sequences)
y = np.array(y_sequences)

# Shuffle sequences
rng = np.random.default_rng(seed=42)
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

# Save
np.save(os.path.join(dataset_path, "X_muse_test.npy"), X)
np.save(os.path.join(dataset_path, "y_muse_test.npy"), y)

print(f"Saved: X shape {X.shape}, y shape {y.shape}")

# print amount of lies and truths in y
print(f"Number of truths: {np.sum(y == 1)}")
print(f"Number of lies: {np.sum(y == 0)}")
