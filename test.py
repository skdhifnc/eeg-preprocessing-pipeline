import h5py
import config

with h5py.File(config.HDF5_VIEW_INPUT_PATH, "r") as f:
    ds = f["RawData/Samples"]
    print(f"Chunks: {ds.chunks}")          # 如果输出 None，说明没分块
    print(f"Compression: {ds.compression}")  # 如果输出 None，说明没压缩