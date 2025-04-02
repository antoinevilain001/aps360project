import os
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
from sklearn.model_selection import train_test_split

# Set dataset directory
dataset_path = "LieWaves"
raw_data_path = os.path.join(dataset_path, "Raw")

# Parameters
channel_names = ['AF3', 'AF4', 'T7', 'T8', 'Pz']
sfreq = 128
window_length = 2  # seconds
overlap = 0.5
nperseg = int(window_length * sfreq)
noverlap = int(nperseg * overlap)
sequence_length = 5  # for LSTM

# Load stimulus-label mappings
stimuli_df = pd.read_excel(os.path.join(dataset_path, "Subject_Stimuli.xlsx"))
file_label_mapping = {
    f"{row['SUBJECT']}{row['SESSION']}.csv": row['LIE/TRUTH']
    for _, row in stimuli_df.iterrows()
}

# Collect all single-channel PSDs and labels
psd_data = []
labels = []

for filename, label in file_label_mapping.items():
    df = pd.read_csv(os.path.join(raw_data_path, filename))
    if 'time' in df.columns[0].lower():
        df = df.iloc[:, 1:]
    data = df.values.T

    # MNE setup and filtering
    info = mne.create_info(channel_names, sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    raw.filter(l_freq=0.5, h_freq=45, fir_design='firwin')
    raw.notch_filter(freqs=[50, 60], fir_design='firwin')
    data = raw.get_data()

    # Segment and extract PSD per channel
    for start in range(0, data.shape[1] - nperseg + 1, nperseg):  # no overlap
        segment = data[:, start:start + nperseg]
        psds, freqs = psd_array_welch(segment, sfreq=sfreq, fmin=0.5, fmax=45,
                                      n_fft=nperseg, n_overlap=0,
                                      window='hamming', average='mean')
        for ch in range(psds.shape[0]):
            psd_data.append(psds[ch][:, np.newaxis])  # (90, 1)
            labels.append(label)

# Convert and reshape into sequences
X = np.array(psd_data)
y = np.array(labels)
num_seq = len(X) // sequence_length
X = X[:num_seq * sequence_length].reshape((num_seq, sequence_length, 90, 1))
y = y[:num_seq * sequence_length].reshape((num_seq, sequence_length))
y = np.array([np.bincount(seq).argmax() for seq in y])  # majority label per sequence

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Save all
np.save(os.path.join(dataset_path, "X_train_seq.npy"), X_train)
np.save(os.path.join(dataset_path, "y_train_seq.npy"), y_train)
np.save(os.path.join(dataset_path, "X_val_seq.npy"), X_val)
np.save(os.path.join(dataset_path, "y_val_seq.npy"), y_val)
np.save(os.path.join(dataset_path, "X_test_seq.npy"), X_test)
np.save(os.path.join(dataset_path, "y_test_seq.npy"), y_test)

print("Sequence-shaped single-channel PSD data saved!")
print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
