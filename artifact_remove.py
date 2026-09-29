# 导入必要的库
from copy import deepcopy  # 用于深度拷贝对象，避免修改原始数据

import h5py  # 用于处理HDF5格式数据文件
import numpy as np  # 数值计算库

import config  # 导入项目配置
from ICA_topomap import load_or_fit_ica  # 导入ICA加载/拟合函数
from eeg_processing import load_or_create_epochs  # 导入epochs加载/创建函数


def crop_epochs_for_csp(epochs):
    """在ICA处理后裁剪epochs，去掉不送入CSP的基线期。"""
    if not config.DROP_BASELINE_BEFORE_CSP:
        return epochs

    cropped_epochs = epochs.copy().crop(
        tmin=config.CSP_EPOCH_TMIN,
        tmax=config.CSP_EPOCH_TMAX,
    )
    print(
        "Cropped epochs for CSP: "
        f"{epochs.times[0]:.3f}s~{epochs.times[-1]:.3f}s -> "
        f"{cropped_epochs.times[0]:.3f}s~{cropped_epochs.times[-1]:.3f}s"
    )
    print(f"CSP input shape: {cropped_epochs.get_data().shape}")
    return cropped_epochs


def artifact_component_remove(epochs, ica, output_path, exclude_components=None):
    """
    使用ICA去除伪影成分，并保存清洗后的EEG数据
    
    参数:
        epochs: MNE-Python的Epochs对象，包含分块后的EEG数据
        ica: 拟合好的ICA对象
        output_path: 清洗后数据的输出路径
        exclude_components: 需要排除的ICA成分索引列表，这些成分通常是伪影
    
    返回:
        epochs_clean: 清洗后的Epochs对象
        ica: 修改后的ICA对象（已设置exclude属性）
    """
    ica = deepcopy(ica)  # 深度拷贝ICA对象，不修改原始ICA
    ica.exclude = exclude_components or []  # 设置需要排除的成分列表
    print(f"Excluded ICA components: {ica.exclude}")

    print("Reconstructing cleaned EEG...")
    epochs_clean = ica.apply(epochs.copy())  # 应用ICA，去除排除的成分，重建干净EEG
    epochs_clean = crop_epochs_for_csp(epochs_clean)  # 去掉基线期后再保存给CSP

    # 提取数据并转换单位为微伏(uV)
    clean_eeg = epochs_clean.get_data() * 1e6  # 将原始数据从V转换为uV
    labels = epochs_clean.events[:, 2]  # 提取每个epoch的标签
    sfreq = epochs_clean.info["sfreq"]  # 提取采样率
    ch_names = epochs_clean.ch_names  # 提取通道名称列表

    # 将清洗后的数据保存为HDF5格式
    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("clean_eeg", data=clean_eeg.astype(np.float32))  # 保存EEG数据
        h5_file.create_dataset("labels", data=labels.astype(np.int32))  # 保存标签
        h5_file.create_dataset("channel_names", data=np.array(ch_names, dtype="S"))  # 保存通道名称
        h5_file.attrs["sampling_rate"] = sfreq  # 保存采样率到文件属性
        h5_file.attrs["unit"] = "microvolts"  # 保存数据单位信息

    # 打印保存信息
    print(f"Saved cleaned EEG to: {output_path}")
    print(f"Saved shape: {clean_eeg.shape}, max amplitude: {np.max(np.abs(clean_eeg)):.2f} uV")
    return epochs_clean, ica


def main():
    """主函数：执行完整的ICA伪影去除流程"""
    # 确保项目所需目录都存在
    config.ensure_project_dirs()
    
    # 加载或创建epochs（分块数据），使用缓存加快处理
    epochs = load_or_create_epochs(
        hdf5_path=config.CONCAT_OUTPUT_PATH,  # 输入拼接好的HDF5文件
        cache_path=config.EPOCHS_CACHE_PATH,  # epochs缓存路径
        force_recompute=config.FORCE_REBUILD_EPOCHS,  # 是否强制重新计算
        show_plots=False,  # 是否显示图表
    )
    
    # 加载或拟合ICA
    prepared_epochs, ica = load_or_fit_ica(
        epochs,  # 输入epochs数据
        cache_path=config.ICA_CACHE_PATH,  # ICA结果缓存路径
        force_recompute=config.FORCE_REBUILD_ICA,  # 是否强制重新拟合
    )
    
    # 执行伪影去除并保存结果
    return artifact_component_remove(
        epochs=prepared_epochs,  # 预处理好的epochs数据
        ica=ica,  # 拟合好的ICA对象
        output_path=config.CLEAN_OUTPUT_PATH,  # 输出文件路径
        exclude_components=config.EXCLUDE_COMPONENTS,  # 需要排除的伪影成分
    )


if __name__ == "__main__":
    main()
