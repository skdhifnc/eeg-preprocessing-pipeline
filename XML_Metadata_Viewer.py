import h5py
import config

with h5py.File(config.ACQUISITION_VIEW_INPUT_PATH, 'r') as f:
    # 尝试读取 AcquisitionTaskDescription，里面通常是采集硬件配置（通道数、采样率、设备型号名称）
    desc = f['RawData/AcquisitionTaskDescription'][0]
    if isinstance(desc, bytes):
        desc = desc.decode('utf-8', errors='ignore')

    asynchron_signal_types = None
    if 'AsynchronData/AsynchronSignalTypes' in f:  #读取AsynchronSignalTypes，里面是事件配置，有关标签的配置
        asynchron_signal_types = f['AsynchronData/AsynchronSignalTypes'][0]
        if isinstance(asynchron_signal_types, bytes):
            asynchron_signal_types = asynchron_signal_types.decode('utf-8', errors='ignore')
        
with open('acquisition_desc.xml', 'w', encoding='utf-8') as out:
    out.write(desc)
print("已保存到 acquisition_desc.xml")

if asynchron_signal_types is not None:
    with open('asynchron_signal_types.xml', 'w', encoding='utf-8') as out:
        out.write(asynchron_signal_types)
    print("已保存到 asynchron_signal_types.xml")
else:
    print("未找到 AsynchronData/AsynchronSignalTypes")