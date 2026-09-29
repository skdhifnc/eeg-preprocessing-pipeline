import h5py
import config

def print_hdf5_structure(name, obj):
    """递归打印文件结构"""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")

# 打开文件
filepath = config.HDF5_VIEW_INPUT_PATH  # 在本地 config.py 中设置输入路径
with h5py.File(filepath, 'r') as f:
    print("文件根目录下的键：", list(f.keys()))
    print("\n完整结构：")
    f.visititems(print_hdf5_structure)
    
    # 读取具体数据示例（假设里面有个 dataset 叫 'eeg_data'）
    if 'eeg_data' in f:
        data = f['eeg_data'][:]
        print(f"\n数据形状: {data.shape}")
        print(f"前5个值: {data.flat[:5]}")