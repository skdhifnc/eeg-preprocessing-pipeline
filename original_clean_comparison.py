import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

import config


N_CHANNELS_TO_SHOW = 32
TIME_WINDOW_SEC = 8
GAIN = 1.0


def find_eeg_dataset(h5_file):
    candidates = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        shape = obj.shape
        if len(shape) == 1:
            return

        lower_name = name.lower()
        score = 0
        for keyword in ["data", "eeg", "signal", "sample", "raw", "clean", "value"]:
            if keyword in lower_name:
                score += 10
        for keyword in ["name", "label", "time", "type", "description", "version", "capabilit"]:
            if keyword in lower_name:
                score -= 20
        if len(shape) == 2:
            score += 5
        if len(shape) >= 2 and (1 <= shape[0] <= 1000 or 1 <= shape[1] <= 1000):
            score += 5

        candidates.append((name, shape, score))

    h5_file.visititems(visitor)
    if not candidates:
        raise ValueError("No suitable EEG dataset was found.")

    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0][0]


def load_hdf5_data(file_path, data_key=None):
    with h5py.File(file_path, "r") as h5_file:
        data_key = data_key or find_eeg_dataset(h5_file)
        data = h5_file[data_key][:]
        return data, data_key


def flatten_if_needed(data):
    if data.ndim == 3:
        return np.transpose(data, (1, 0, 2)).reshape(data.shape[1], -1)
    if data.ndim > 3:
        return data.reshape(data.shape[0], -1)
    return data


def prepare_data(raw_data, clean_data):
    raw_data = flatten_if_needed(raw_data)
    clean_data = flatten_if_needed(clean_data)

    if raw_data.ndim != 2 or clean_data.ndim != 2:
        raise ValueError("Both datasets must be convertible to 2D channel x time arrays.")

    if raw_data.shape[0] > raw_data.shape[1] * 2:
        raw_data = raw_data.T
    if clean_data.shape[0] > clean_data.shape[1] * 2:
        clean_data = clean_data.T

    min_channels = min(raw_data.shape[0], clean_data.shape[0])
    min_samples = min(raw_data.shape[1], clean_data.shape[1])
    raw_data = raw_data[:min_channels, :min_samples]
    clean_data = clean_data[:min_channels, :min_samples]
    return raw_data, clean_data, min_channels, min_samples


class EEGComparator:
    def __init__(self, raw_data, clean_data, sfreq):
        self.raw = raw_data
        self.clean = clean_data
        self.sfreq = sfreq
        self.n_channels, self.n_samples = raw_data.shape
        self.offsets = np.arange(self.n_channels) * 3
        self.start_sample = 0
        self.window_samples = int(TIME_WINDOW_SEC * sfreq)
        self.scale = GAIN
        self.visible_channels = min(N_CHANNELS_TO_SHOW, self.n_channels)
        self._setup_figure()
        self._update_plot()

    def _setup_figure(self):
        self.fig, (self.ax_raw, self.ax_clean) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        self.fig.patch.set_facecolor("black")

        for ax, title in [
            (self.ax_raw, "RAW DATA"),
            (self.ax_clean, "CLEAN DATA"),
        ]:
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.set_title(title, color="white", fontsize=14, fontweight="bold")
            ax.set_ylabel("Channels", color="white")
            ax.grid(True, alpha=0.3, color="gray")

        self.ax_clean.set_xlabel("Time (s)", color="white")
        self.lines_raw = []
        self.lines_clean = []
        for idx in range(self.n_channels):
            color = plt.cm.tab20(idx % 20)
            line_raw, = self.ax_raw.plot([], [], color=color, linewidth=0.8)
            line_clean, = self.ax_clean.plot([], [], color=color, linewidth=0.8)
            self.lines_raw.append(line_raw)
            self.lines_clean.append(line_clean)

        self.fig.subplots_adjust(bottom=0.15)
        ax_slider = self.fig.add_axes([0.2, 0.02, 0.6, 0.03], facecolor="gray")
        self.slider = Slider(
            ax_slider,
            "Time",
            0,
            max(0, self.n_samples - self.window_samples),
            valinit=0,
            color="cyan",
        )
        self.slider.on_changed(self._on_slider_change)

        ax_zoom_in = self.fig.add_axes([0.85, 0.02, 0.05, 0.03])
        self.btn_zoom_in = Button(ax_zoom_in, "Zoom+", color="gray", hovercolor="lightblue")
        self.btn_zoom_in.on_clicked(self._zoom_in)

        ax_zoom_out = self.fig.add_axes([0.91, 0.02, 0.05, 0.03])
        self.btn_zoom_out = Button(ax_zoom_out, "Zoom-", color="gray", hovercolor="lightblue")
        self.btn_zoom_out.on_clicked(self._zoom_out)

        ax_gain_up = self.fig.add_axes([0.05, 0.02, 0.05, 0.03])
        self.btn_gain_up = Button(ax_gain_up, "Gain+", color="gray", hovercolor="lightblue")
        self.btn_gain_up.on_clicked(self._gain_up)

        ax_gain_down = self.fig.add_axes([0.11, 0.02, 0.05, 0.03])
        self.btn_gain_down = Button(ax_gain_down, "Gain-", color="gray", hovercolor="lightblue")
        self.btn_gain_down.on_clicked(self._gain_down)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

    def _get_time_axis(self):
        start = self.start_sample / self.sfreq
        end = (self.start_sample + self.window_samples) / self.sfreq
        return np.linspace(start, end, self.window_samples)

    def _update_plot(self):
        end_sample = min(self.start_sample + self.window_samples, self.n_samples)
        actual_window = end_sample - self.start_sample
        time_axis = self._get_time_axis()[:actual_window]

        for idx in range(self.n_channels):
            if idx < self.visible_channels:
                raw_segment = self.raw[idx, self.start_sample:end_sample]
                clean_segment = self.clean[idx, self.start_sample:end_sample]
                self.lines_raw[idx].set_data(time_axis, raw_segment * self.scale + self.offsets[idx])
                self.lines_clean[idx].set_data(time_axis, clean_segment * self.scale + self.offsets[idx])
                self.lines_raw[idx].set_visible(True)
                self.lines_clean[idx].set_visible(True)
            else:
                self.lines_raw[idx].set_visible(False)
                self.lines_clean[idx].set_visible(False)

        y_limit = self.offsets[self.visible_channels - 1] + 5 if self.visible_channels > 0 else 10
        for ax in [self.ax_raw, self.ax_clean]:
            ax.set_xlim(time_axis[0], time_axis[-1])
            ax.set_ylim(-2, y_limit)
            ax.set_yticks(self.offsets[:self.visible_channels])
            ax.set_yticklabels([f"Ch{i + 1}" for i in range(self.visible_channels)])

        self.fig.canvas.draw_idle()

    def _on_slider_change(self, value):
        self.start_sample = int(value)
        self._update_plot()

    def _zoom_in(self, _event=None):
        self.window_samples = max(int(self.window_samples * 0.8), 100)
        self._update_plot()

    def _zoom_out(self, _event=None):
        self.window_samples = min(int(self.window_samples * 1.2), self.n_samples)
        if self.start_sample + self.window_samples > self.n_samples:
            self.start_sample = max(0, self.n_samples - self.window_samples)
            self.slider.set_val(self.start_sample)
        self._update_plot()

    def _gain_up(self, _event=None):
        self.scale *= 1.2
        self._update_plot()

    def _gain_down(self, _event=None):
        self.scale *= 0.8
        self._update_plot()

    def _on_key(self, event):
        if event.key == "right":
            step = int(self.window_samples * 0.1)
            self.start_sample = min(self.start_sample + step, self.n_samples - self.window_samples)
            self.slider.set_val(self.start_sample)
        elif event.key == "left":
            step = int(self.window_samples * 0.1)
            self.start_sample = max(self.start_sample - step, 0)
            self.slider.set_val(self.start_sample)
        elif event.key == "up":
            self.visible_channels = min(self.visible_channels + 1, self.n_channels)
            self._update_plot()
        elif event.key == "down":
            self.visible_channels = max(self.visible_channels - 1, 1)
            self._update_plot()
        elif event.key == "+":
            self._gain_up()
        elif event.key == "-":
            self._gain_down()

    def show(self):
        print("Controls:")
        print("- Slider: move through time")
        print("- Zoom+/Zoom-: change time window")
        print("- Gain+/Gain-: change amplitude scaling")
        print("- Left/Right: move the window")
        print("- Up/Down: change visible channel count")
        plt.show()


def compare_raw_clean(raw_path=None, clean_path=None, sfreq=None):
    raw_path = raw_path or config.RAW_COMPARISON_PATH
    clean_path = clean_path or config.CLEAN_COMPARISON_PATH
    sfreq = sfreq or config.DEFAULT_SFREQ

    raw_data, _ = load_hdf5_data(raw_path)
    clean_data, _ = load_hdf5_data(clean_path, data_key="clean_eeg")
    raw_data, clean_data, n_ch, n_samples = prepare_data(raw_data, clean_data)

    print(f"Prepared comparison data: {n_ch} channels x {n_samples} samples")
    viewer = EEGComparator(raw_data, clean_data, sfreq)
    viewer.show()
    return viewer


def main():
    return compare_raw_clean()


if __name__ == "__main__":
    main()
