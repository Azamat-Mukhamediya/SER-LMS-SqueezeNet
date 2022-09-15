import librosa
import numpy as np
from torch.utils.data import Dataset


class SERDataset(Dataset):
    def __init__(self, dataset, dataset_type, transform=None, target_transform=None):
        self.dataset_size = len(dataset)
        self.dataset_type = dataset_type

        self.transform = transform
        self.target_transform = target_transform
        spectrograms = []
        labels = []

        for data, label_id in dataset:
            data = np.expand_dims(data, -1)

            spectrograms.append(data)

            labels.append(np.int64(label_id))

        self.data = spectrograms
        self.labels = np.array(labels)

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, idx):
        X = self.data[idx]
        Y = self.labels[idx]
        if self.transform:
            X = self.transform(X)

        return X, Y


def get_spectrogram(waveform, win_len, hop_len, n_mel):

    waveform = np.array(waveform, dtype='f')
    spectrogram = librosa.feature.melspectrogram(
        waveform, sr=16000, n_mels=n_mel, n_fft=win_len, hop_length=hop_len, win_length=win_len)
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram


def get_audio_segmentation(data, seg_len, win_len, hop_len, n_mel):
    segmented_data = []
    segmented_labels = []

    for audio, label_id in data:
        if(audio.shape[0] < seg_len):
            zr = seg_len - audio.shape[0]
            zeros = np.zeros(zr, dtype=int)
            audio = np.concatenate((audio, zeros))

        frames = librosa.util.frame(
            audio, frame_length=seg_len, hop_length=seg_len, axis=0)

        for frame in frames:
            spectrogram = get_spectrogram(
                frame, win_len=win_len, hop_len=hop_len, n_mel=n_mel)

            audio_with_label = [spectrogram, label_id]

            segmented_data.append(audio_with_label)
            segmented_labels.append(label_id)

    return segmented_data, segmented_labels
