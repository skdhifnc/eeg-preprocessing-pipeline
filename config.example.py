"""EEG 预处理流水线公开配置模板。

首次使用时复制本文件为同目录下的 config.py，再填写自己的数据路径。
已有 config.py 时不要覆盖，以免丢失本机配置。本模板不会自动被流水线读取。
输入编号仅为示例；算法参数、通道布局和时间窗须按自己的实验审核。
ICA 排除列表默认留空，发布确认默认关闭，不携带任何已有实验的审核结论。

配置服务于阶段 1/2 预处理及拼接、阶段 3 人工 ICA 审核和阶段 4 分段导出。

阅读顺序：输入文件 -> 阶段 1 滤波/坏段 -> 阶段 2 拼接/参考 -> ICA -> Epoch 导出。
本文件只定义参数，不执行数据处理。修改后保存，再运行对应阶段的入口脚本。
True/False 是布尔开关；None 表示不指定或不启用，含义以各参数说明为准。
() 是空元组；单个元素写成 (21,) 或 ("Fp1",)，末尾逗号不可省略。
上游参数变化后需要重跑相关阶段，旧模型/预览/导出不会自动更新。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent  # 项目目录：由本配置文件位置确定，不依赖终端当前目录。
DATASET_DIR = PROJECT_ROOT / "data"  # 示例输入目录；在本地 config.py 中改为自己的数据目录，不公开真实数据。

# 辅助查看/转换脚本的示例输入；复制为 config.py 后按自己的数据填写。
ACQUISITION_VIEW_INPUT_PATH = DATASET_DIR / "1-10s-S.hdf5"
HDF5_VIEW_INPUT_PATH = DATASET_DIR / "1-10s-S.hdf5"
MAT_DATASET_DIR = DATASET_DIR
MAT_INPUT_PATH = MAT_DATASET_DIR / "dataset1_processeddata_Sub-001_sess-01_task-MI_eeg.mat"
MAT_OUTPUT_PATH = MAT_DATASET_DIR / "clean_dataset1_processeddata_Sub-001_sess-01_task-MI_eeg.hdf5"
MAT_BATCH_START_SUB = 1
MAT_BATCH_END_SUB = 1
MAT_BATCH_N_SESSIONS = 1

# 批量输入配置。INPUT_INDICES 非空时优先使用；为空时使用闭区间
# INPUT_INDEX_START ~ INPUT_INDEX_END，并排除 EXCLUDED_INPUT_INDICES。
INPUT_FILE_TEMPLATE = "{index}-10s-{condition}.hdf5"  # 文件名模板；两个占位符分别替换为编号和条件字符串。
INPUT_CONDITION = "S"  # 文件名中的 condition；这里只用于选文件，不决定 left/right 分类标签。
INPUT_INDICES: tuple[int, ...] = ()  # 显式编号，如 (21, 23)；非空时优先于起止范围，按填写顺序拼接。
INPUT_INDEX_START = 1  # INPUT_INDICES 为空时的起始编号，包含该编号。
INPUT_INDEX_END = 10  # INPUT_INDICES 为空时的结束编号，包含该编号；不得小于起始值。
EXCLUDED_INPUT_INDICES: tuple[int, ...] = ()  # 从上述任一种编号选择中排除的编号，如 (22,)。
VALIDATE_INPUT_FILES = True  # True：生成路径时集中检查文件存在；False 仅跳过预检查，读取时仍须存在。

# HDF5 数据与元数据。
SOURCE_DATA_UNIT = "microvolts"  # 原始数值的实际单位；微伏读入后乘 1e-6 转为 MNE 使用的伏，不能凭幅值猜单位。
DEFAULT_SFREQ = 250.0  # 元数据缺失且允许回退时使用的采样率，单位 Hz；不是重采样目标。
ALLOW_METADATA_FALLBACK = False  # False：元数据问题报错；True 可回退采样率/生成 Ch1 等名称，但后续电极校验仍可能拒绝。
ACTIVE_EVENT_VALUE = 1  # 只保留 AsynchronData/Value 等于该值的有效事件，不是 left/right 的类别编码。
# 部分采集文件在记录末尾包含晚于最后一个 EEG 样本的事件。此类事件
# 无法成为有效 MNE Annotation；默认跳过并警告，设为 False 可改为报错。
SKIP_OUT_OF_RANGE_EVENTS = True

# 阶段 1 基础滤波：仅做 1 Hz 高通，不做平均参考和低通。
HIGH_PASS_FREQ = 1.0  # 高通下限，单位 Hz，用于减少慢漂移；后续 ICA 要求输入至少做过 1 Hz 高通。
HIGH_PASS_METHOD = "iir"  # 高通实现方式；当前调用配套传入 IIR 参数，不建议孤立改成 FIR。
IIR_ORDER = 4  # IIR 设计阶数；阶数越高过渡越陡，但时域响应也会变化。
IIR_FILTER_TYPE = "butter"  # IIR 类型：Butterworth（巴特沃斯），通带幅频响应平坦。

# annotate_amplitude 检测相邻采样点的突变，而非信号绝对幅值。
# bad_percent=100 使异常优先作为时间段注释返回，而不是转为坏通道。
ANNOTATE_AMPLITUDE_PEAK_UV = 100.0  # 相邻采样点电压差的突变阈值，单位微伏；不是 abs(EEG) 阈值。
ANNOTATE_AMPLITUDE_BAD_PERCENT = 100.0  # 通道异常占比阈值（%）；保持 100，让 MNE 返回坏时间段而非坏通道。
ANNOTATE_AMPLITUDE_MIN_DURATION = 0.004  # 突变异常最短持续时间，单位秒；250 Hz 时 0.004 s 为一个采样间隔。

# 真正的绝对幅值检测：任一 EEG 通道满足 abs(signal) > 阈值时标记。
ABSOLUTE_AMPLITUDE_THRESHOLD_UV = 300.0  # 高通后绝对电压阈值，单位微伏；任一通道超限则该时间点异常。
ABSOLUTE_AMPLITUDE_MIN_DURATION = 0.004  # 绝对幅值连续超限需达到的时长，单位秒；更短片段不标记。

# 两类异常区间合并后使用统一名称。名称必须以 bad 开头，才能被
# mne.Epochs(reject_by_annotation=True) 自动拒绝。
PREICA_BAD_AMPLITUDE_LABEL = "bad_preica_amplitude"  # 初筛：保护坏通道检测和 ICA fit，不直接决定最终拒绝。
BAD_AMPLITUDE_LABEL = "bad_amplitude"  # ICA 后复检的最终幅值标记；旧阶段 1 同名标记在复检时精确替换。
BAD_AMPLITUDE_MERGE_GAP_SECONDS = 0.0  # 合并坏段允许跨过的最大间隔，单位秒；0 仅合并重叠或首尾相接区间。

# 阶段 1 质量检查。这里只验证 bad_preica_amplitude 能拒绝受污染 epoch；
# 阶段 2 的实际输入仍是连续 Raw 列表，最终分段将在 ICA 后完成。
RUN_EPOCH_REJECTION_CHECK = True  # 是否在每个短 Raw 上试切 Epoch 并打印拒绝数量；不会裁剪/替换正式 Raw。
QC_EPOCH_TMIN = -1.0  # 质量检查窗口起点，单位秒，相对事件开始；负值表示事件前。
QC_EPOCH_TMAX = 4.0  # 质量检查窗口终点，单位秒，MNE 包含终点；越界 Epoch 也会被拒绝。
QC_EPOCH_BASELINE = (-0.2, 0.0)  # 仅 QC Epoch 的基线区间（秒）；减去该区间均值，None 可关闭，不影响连续 Raw。

# 阶段 2：固定路径供独立 ICA 脚本重新加载最近一次成功发布的结果。
# 临时文件在同一目录下的唯一临时子目录内写入，成功回读后原子替换。
CONCATENATED_RAW_PATH = PROJECT_ROOT / "cache" / "stage2_concatenated_raw.fif"  # 阶段 2 正式长 Raw 输出，也是 ICA 输入。
CONCATENATED_RAW_OVERWRITE = True  # True：验证成功后替换旧阶段 2 文件；False：已有文件时报错，不覆盖。
CONCATENATION_ON_MISMATCH = "raise"  # MNE 拼接元数据不一致时的策略；当前流程只允许 raise，禁止静默忽略。
CONCATENATED_RAW_FORMAT = "double"  # FIF 浮点精度：double 为 64 位，single 为 32 位；不改变 EEG 的伏单位。
BAD_BOUNDARY_LABEL = "bad_boundary"  # 拼接接缝名；阶段 2 强制该值，零时长注释用于阻止跨试段 Epoch。
# 删除 EDGE boundary 后，后续滤波必须显式使用这个 skip_by_annotation。
CONTINUOUS_FILTER_SKIP_ANNOTATIONS = ("edge", "bad_acq_skip", BAD_BOUNDARY_LABEL)  # 后续连续滤波的边界约定；仅定义不会自动生效，调用 filter 时须传入。阶段 4 使用自己的扩展列表。

# 拼接后：识别持续性坏通道 -> 插值 -> 即时平均参考。阈值须结合报告审核。
BAD_CHANNEL_LOF_N_NEIGHBORS = 20  # 局部离群因子 LOF 比较的邻居通道数；实际不超过候选数减 1，候选不足 4 时跳过 LOF。
BAD_CHANNEL_LOF_THRESHOLD = 1.5  # LOF 异常分数阈值，必须 >1；越低通常越容易标坏，须结合通道质量审核。
BAD_CHANNEL_FLAT_STD_UV = 0.01  # 可用样本上的通道标准差 <= 此值视为近乎平线，单位微伏。
BAD_CHANNEL_MIN_STD_RATIO = 0.01  # 通道标准差 / 全部通道标准差中位数的下限；过低可能提示接触/信号异常。
BAD_CHANNEL_MAX_STD_RATIO = 20.0  # 上述标准差比值的上限；过高可能提示持续噪声，必须 >1。
BAD_CHANNEL_MIN_GOOD_SECONDS = 2.0  # 去掉 bad* 样本后至少剩余的总时长，单位秒；不足则停止检测，不是每段都需 2 秒。
BAD_CHANNEL_BOUNDARY_PADDING_SECONDS = 1.0  # 仅在检测/秩估计副本中向接缝前后各扩展的秒数，避开边缘影响。
BAD_CHANNEL_MAX_COUNT = 4  # 允许修复的最大坏通道数（含人工指定）；超出则报错要求审核，不直接删除通道。
BAD_CHANNEL_MIN_GOOD_CHANNELS = 4  # 有坏通道需插值时，至少保留的好通道供体数；达到此数不保证插值可靠。
# 使用标准电极名称；INCLUDE 强制加入，EXCLUDE 撤销自动/已有标记，二者不可重叠。
BAD_CHANNELS_MANUAL_INCLUDE: tuple[str, ...] = ()  # 人工强制标坏并插值的标准通道名，如 ("Fp1",)；不是 ICA 成分编号。
BAD_CHANNELS_MANUAL_EXCLUDE: tuple[str, ...] = ()  # 从自动/已有坏通道集合撤销的名称；此处 EXCLUDE 是“取消标坏”，不是删除通道。
# 相对于最大奇异值的容差，抑制插值舍入残差；不会向数据添加噪声。
EEG_RANK_REL_TOL = 1e-6  # 有效秩的相对阈值：仅统计 > 最大奇异值×此值的维度；越大估计越保守，必须在 (0,1)。

# 阶段 3：独立脚本 ICA_processing.py。不要并发运行 fit/preview/apply。
# VS Code 按钮入口：保存本配置，再打开 ICA_processing.py 点“运行 Python 文件”。
# 每次只执行一个阶段；显式命令行参数优先于下面的运行控制配置。
ICA_RUN_MODE = "fit"  # "fit" / "review"（或 "inspect"）/ "preview" / "apply"
ICA_FORCE_REFIT = False  # 仅 fit：True 强制重拟合，旧审核/预览随之失效
ICA_SHOW_PLOTS = True  # review/preview：False 只导出图片，不弹出窗口
# 仅 review：None 查看 GUI 标记的候选；() 不看属性图；(0, 2) 查看指定编号。
# 这不是正式排除列表，也不代表建议排除任何成分；按钮运行不要求终端输入。
ICA_REVIEW_COMPONENTS: tuple[int, ...] | None = None
ICA_APPLY_CONFIRM = False  # 仅 apply：人工检查预览后才设 True，发布后请改回 False

ICA_INPUT_RAW_PATH = CONCATENATED_RAW_PATH  # ICA 唯一 EEG 输入：阶段 2 已插值且即时平均参考的连续 Raw。
ICA_MODEL_PATH = PROJECT_ROOT / "cache" / "stage3-ica.fif"  # 保存分解模型，不是清洗后的 EEG；fit 生成，review/preview/apply 复用。
ICA_MANIFEST_PATH = PROJECT_ROOT / "cache" / "stage3_ica_manifest.json"  # 模型清单：记录输入哈希、拟合参数、模型 ID/哈希，防止旧模型误用。
ICA_CLEAN_RAW_PATH = PROJECT_ROOT / "cache" / "stage3_ica_clean_raw.fif"  # 仅 apply 发布的重建 EEG；保留全长时间轴和坏段注释。
ICA_REPORT_DIR = PROJECT_ROOT / "reports" / "ica_review"  # 审核图、候选、预览和发布凭据根目录，按每次模型 ID 分子目录。
ICA_MONTAGE = "standard_1020"  # MNE 标准电极坐标模板，供通道定位、插值和头皮图使用；不是实测个体电极位置。
# 按设备 1..32 号排列；旧 ICA_topomap 的 CHANNEL_MAPPING 已逐项核对。
# 设备原始标签为 cz, 2, ..., 32；只有完整设备布局才启用此表。
# ICA_DEVICE_CHANNELS：第 i 项对应设备第 i+1 个电极的标准名称；也用于 EEG1..EEG32 布局识别，禁止随意排序。
ICA_DEVICE_CHANNELS = (
    "Fp1", "Fp2", "FC5", "F8", "Fz", "F3", "FCz", "F4",
    "Oz", "FC3", "T7", "CP6", "FC4", "F7", "C3", "Cz",
    "C4", "FC6", "CP5", "C1", "C2", "T8", "P7", "CP3",
    "CPz", "CP4", "P8", "O1", "P3", "P4", "O2", "Pz",
)
ICA_CLEAN_OVERWRITE = True  # True 允许成功 apply 替换旧 clean；False 遇到旧文件报错，不控制 fit 模型覆盖。
ICA_N_COMPONENTS = 15  # ICA 独立成分个数，整数 >=2，不能超过可用有效秩；不是输出 EEG 通道数。
ICA_METHOD = "fastica"  # 分解算法；当前实现只接受已验证的 FastICA，改其他算法会被拒绝。
ICA_RANDOM_STATE = 42  # 随机初始化种子；同输入和环境下便于复现，换种子后需重新拟合/审核。
ICA_MAX_ITER = "auto"  # 最大迭代数：auto 交由 MNE 按算法设置，也可填正整数；未收敛时不发布模型。
ICA_DECIM = None  # 当前不抽点，避免仅高通数据抽点产生混叠。
ICA_REJECT_BY_ANNOTATION = True  # 拟合时省略 bad* 样本；当前必须 True，不会从正式 Raw 中删除这些样本。
ICA_BOUNDARY_PADDING_SECONDS = 1.0  # 只在拟合副本中扩展边界
ICA_EXCLUDE_COMPONENTS: tuple[int, ...] = ()  # 0-based: ICA000 对应 0；新模型必须重新审核
ICA_SOURCE_WINDOW_SECONDS = 20.0  # 源浏览器初始窗口及前后预览窗口长度，单位秒；不决定拟合数据长度或最终 Epoch 长度。
ICA_SOURCE_VISIBLE_COMPONENTS = 15  # 源时序浏览器每页显示的成分数，正整数；不改变实际 ICA 成分总数。
ICA_PSD_FMIN = 1.0  # 成分属性图功率谱显示下限，单位 Hz；只影响可视化，不执行高通滤波。
ICA_PSD_FMAX = 60.0  # 属性图功率谱显示上限，单位 Hz，实际不超过 Nyquist=采样率/2；不是低通截止频率。
ICA_PREVIEW_CHANNELS = ("Fp1", "Fp2", "C3", "Cz", "C4")  # 前后对比波形优先展示的 EEG 通道；不限制重建/保存通道。

# ICA 后、最终带通前复检；独立于阶段 1 配置。修改后须重新 preview/apply，不必仅因此重 fit。
POSTICA_AMPLITUDE_PEAK_UV = 100.0  # 相邻点突变阈值（微伏），不是绝对幅值。
POSTICA_AMPLITUDE_PEAK_MIN_DURATION = 0.004  # 突变最短时长（秒）。
POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV = 300.0  # 任一 EEG 通道绝对幅值阈值（微伏）。
POSTICA_ABSOLUTE_AMPLITUDE_MIN_DURATION = 0.004  # 绝对幅值超限最短时长（秒）。
POSTICA_AMPLITUDE_MERGE_GAP_SECONDS = 0.0  # 合并异常区间的最大间隔（秒）。
# 只撤销精确的幅值初筛标记；其他 bad*/edge* 一律保留。
# 硬件/数据损坏请使用独立名称，如 bad_dropout、bad_clipping、bad_flat，不能使用幅值标记名。

# 阶段 4：阶段 3 clean Raw -> 连续带通 -> 事件分段 -> 双 HDF5 输出。
# 时间窗、滤波、基线、单位和浮点类型都由这里控制；实现会按实际采样率自动确定采样点数。
STAGE4_INPUT_RAW_PATH = ICA_CLEAN_RAW_PATH  # 阶段 3 ICA clean Raw；为保证来源可追溯，不回退读取阶段 2 EEG。
STAGE4_FULL_OUTPUT_PATH = PROJECT_ROOT / "cache" / "stage4_epochs_full.hdf5"  # 完整审核 Epoch 输出，长度由 STAGE4_EPOCH_TMIN/TMAX 决定。
STAGE4_WS_OUTPUT_PATH = PROJECT_ROOT / "cache" / "stage4_epochs_ws.hdf5"  # 分类输入，从完整 Epoch 裁出，长度由 STAGE4_WS_TMIN/TMAX 决定。
STAGE4_REPORT_PATH = PROJECT_ROOT / "reports" / "stage4_epoch_report.json"  # JSON 审核报告：类别数量、拒绝原因、来源和输出哈希。
STAGE4_OVERWRITE = True  # 是否允许替换旧的两份 HDF5 和报告；False 时任一输出已存在就停止。
STAGE4_FILTER_L_FREQ = 8.0  # 连续 EEG 带通下限，单位 Hz，必须大于 0 且小于上限。
STAGE4_FILTER_H_FREQ = 30.0  # 连续 EEG 带通上限，单位 Hz，还必须小于输入数据的 Nyquist 频率。
STAGE4_FILTER_METHOD = "fir"  # 传给 MNE Raw.filter 的滤波方法。
STAGE4_FILTER_PHASE = "zero"  # 传给 MNE Raw.filter 的相位设置。
STAGE4_FILTER_DESIGN = "firwin"  # FIR 系数设计方法；选择其他方法时须符合当前 MNE 接口要求。
# 滤波跳过的注释前缀；保留坏段和拼接边界，避免滤波跨越它们传播。
STAGE4_FILTER_SKIP_ANNOTATIONS = (
    "bad", "edge", "bad_acq_skip", BAD_BOUNDARY_LABEL, BAD_AMPLITUDE_LABEL,
)
STAGE4_EVENT_ID = {"left": 1, "right": 2}  # 恰好两个分类事件；名称和两个不同的整数标签均可修改。
STAGE4_EPOCH_TMIN = 0.0  # 完整审核窗口起点，单位秒，相对于事件时刻。
STAGE4_EPOCH_TMAX = 5.0  # 完整审核窗口终点，单位秒；采样点数按实际采样率自动确定。
STAGE4_BASELINE = None  # None 关闭基线校正；也可填位于完整窗口内的二元区间，如 (-0.2, 0.0)。
STAGE4_WS_TMIN = 0.0  # 分类窗口起点，必须位于完整审核窗口内。
STAGE4_WS_TMAX = 4.0  # 分类窗口终点；实际边界会对齐到输入数据的采样点。
STAGE4_OUTPUT_UNIT = "microvolts"  # 支持 volts、millivolts、microvolts 或 nanovolts。
STAGE4_OUTPUT_DTYPE = "float32"  # 任意 NumPy 浮点类型，例如 float32 或 float64。

# FBCSP：从同一份阶段 3 ICA clean 连续 Raw 生成的独立四维分类输入。
# 每个频带仍在分类阶段的训练折内独立拟合 CSP；这里仅做无标签滤波和 Epoch 导出。
FBCSP_BANDS: tuple[tuple[float, float], ...] = (
    (8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
    (20.0, 24.0), (24.0, 30.0),
)
FBCSP_FULL_OUTPUT_PATH = PROJECT_ROOT / "cache" / "stage4_fbcsp_epochs_full.hdf5"
FBCSP_WS_OUTPUT_PATH = PROJECT_ROOT / "cache" / "stage4_fbcsp_epochs_ws.hdf5"
FBCSP_REPORT_PATH = PROJECT_ROOT / "reports" / "stage4_fbcsp_epoch_report.json"
FBCSP_BASELINE_OUTPUT_PATH = PROJECT_ROOT / "cache" / "stage4_fbcsp_baseline_ws.hdf5"
FBCSP_OVERWRITE = False  # 显式允许后才覆盖新实验输入；绝不覆盖原 CSP 产物。



def get_input_indices() -> tuple[int, ...]:
    """返回最终输入编号元组：先选显式列表/闭区间，再排除，最后拒绝空集和重复。

    保留用户填写顺序，不自动排序；返回顺序就是后续文件读取与 Raw 拼接顺序。
    """

    if INPUT_INDICES:
        indices = INPUT_INDICES
    else:
        if INPUT_INDEX_END < INPUT_INDEX_START:
            raise ValueError("INPUT_INDEX_END must be greater than or equal to INPUT_INDEX_START")
        indices = tuple(range(INPUT_INDEX_START, INPUT_INDEX_END + 1))

    excluded = set(EXCLUDED_INPUT_INDICES)
    selected = tuple(index for index in indices if index not in excluded)
    if not selected:
        raise ValueError("No input indices remain after applying the input configuration")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate input indices are not allowed: {selected}")
    return selected


def get_input_hdf5_paths(validate: bool | None = None) -> list[Path]:
    """返回按编号顺序生成的 Path 列表，不读取 EEG 内容。

    validate=None 遵从 VALIDATE_INPUT_FILES；显式 True/False 仅覆盖本次路径检查。
    """

    paths = [
        DATASET_DIR
        / INPUT_FILE_TEMPLATE.format(index=index, condition=INPUT_CONDITION)
        for index in get_input_indices()
    ]

    should_validate = VALIDATE_INPUT_FILES if validate is None else validate
    if should_validate:
        missing = [path for path in paths if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Configured HDF5 input files do not exist:\n{formatted}")
    return paths
