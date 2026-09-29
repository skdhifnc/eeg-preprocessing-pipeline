import h5py
import numpy as np
import config

filepath = config.ACQUISITION_VIEW_INPUT_PATH

with h5py.File(filepath, 'r') as f:
    # 读取原始脑电数据 (30003, 32)
    eeg_data = f['RawData/Samples'][:]
    print(f"脑电数据形状: {eeg_data.shape}")  # (30003, 32)
    print(f"数据类型: {eeg_data.dtype}")      # float32
    
    # 读取事件标记（如果有）
    if 'AsynchronData/Time' in f:
        event_time = f['AsynchronData/Time'][:]
        event_type = f['AsynchronData/TypeID'][:]
        print(f"事件时间点: {event_time.flatten()}")
        print(f"事件类型ID: {event_type.flatten()}")
    
    # 读取元数据（字符串需要解码）
    if 'RawData/SessionDescription' in f:
        session_desc = f['RawData/SessionDescription'][0]
        if isinstance(session_desc, bytes):
            session_desc = session_desc.decode('utf-8', errors='ignore')
        print(f"Session: {session_desc}")