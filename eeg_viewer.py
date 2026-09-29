import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np

import config


def load_raw_samples(file_path, sfreq=config.DEFAULT_SFREQ):
    with h5py.File(file_path, "r") as h5_file:
        samples_data = h5_file["RawData/Samples"][:]
    eeg_data = samples_data.T
    ch_names = [f"Ch{i + 1}" for i in range(eeg_data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(eeg_data, info), eeg_data, ch_names


def view_eeg(file_path=None, sfreq=config.DEFAULT_SFREQ):
    file_path = file_path or config.CONCAT_OUTPUT_PATH
    raw, eeg_data, ch_names = load_raw_samples(file_path, sfreq=sfreq)
    raw.filter(l_freq=0.5, h_freq=30)
    raw.plot(
        duration=10,
        n_channels=len(ch_names),
        scalings="auto",
        start=0,
        title="Continuous EEG viewer",
        show_scrollbars=True,
        show_scalebars=True,
    )

    n_samples = eeg_data.shape[1]
    total_time = n_samples / sfreq
    screen_duration = 10
    n_screens = int(np.ceil(total_time / screen_duration))

    for idx in range(n_screens):
        start_sample = int(idx * screen_duration * sfreq)
        end_sample = int(min((idx + 1) * screen_duration * sfreq, n_samples))
        current_data = eeg_data[:, start_sample:end_sample]
        plt.figure(figsize=(12, 8))
        for channel_idx, (ch_data, ch_name) in enumerate(zip(current_data, ch_names)):
            plt.plot(ch_data + channel_idx * 50, label=ch_name)
        plt.xlabel(f"Time window {idx + 1}")
        plt.ylabel("Channels")
        plt.title(f"Screen {idx + 1}/{n_screens}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def main():
    view_eeg()


if __name__ == "__main__":
    main()
