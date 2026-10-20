# BCI-2.0 EEG 处理流水线

> 公开版说明：本文与 `config.example.py` 配套，不包含作者的本机绝对路径、
> 实验输入编号、ICA 人工排除结论或真实运行目录标识。配置模板不包含 EEG 数据。
> `config.py` 为使用者在本地创建的配置；已有该文件时不要用模板覆盖。
> 本文中的文件路径均相对于项目根目录，命令请在项目根目录的终端中执行。
> `bat` 命令块采用 Windows CMD 语法；运行前先选择或激活已安装依赖的 Python 环境。
> 原始数据、个人配置、采集 XML、缓存和实验报告不随源码发布，由 `.gitignore` 排除。

本项目将采集设备导出的原始 HDF5 EEG 数据，处理为经过人工 ICA 审核、完成
坏试次（Epoch）质量控制的分类输入数据，并提供单频带 CSP-SVM、多频带
FBCSP-SVM、嵌套交叉验证与 Epoch 标签置换检验。

当前 FBCSP 方法为 **grouped-concat 内层搜索**：保留 MNE `cov_est="concat"`，
在训练折内复用同一正则化参数的 CSP 拟合，不使用磁盘变换缓存；默认网格为
**12 个候选、外层 10 折 × 内层 5 折**。完整操作说明见 [FBCSP 专项指南](FBCSP_GUIDE.md)，
快速运行与置换命令见第 9.2 节。

项目的正式数据链路如下：

```text
原始 HDF5
    │
    ├─ 阶段 1：读取元数据、事件和 EEG；1 Hz 高通；标记幅值坏段；质量检查
    │          输出：内存中的短 Raw 列表及其 QC 结果
    │
    ├─ 阶段 2：拼接短 Raw；修复持续性坏通道；即时平均参考；发布连续 Raw FIF
    │          输出：cache/stage2_concatenated_raw.fif
    │
    ├─ 阶段 3：ICA 拟合 → 人工审核 → 预览 → 正式应用
    │          输出：ICA 模型、清单、审核凭据和 clean Raw FIF
    │
    └─ 阶段 4（两条独立分支，均读取同一份 ICA clean Raw）
         ├─ EEG_processing.py：连续 8–30 Hz 带通；Epoch QC；双 HDF5 和报告
         │    └─ main.py → 单频带 CSP-SVM → permutation.py
         └─ fbcsp_stage4.py：五个子带与同源宽带 baseline；对齐 Epoch；独立发布
              └─ fbcsp_main.py → FBCSP-SVM + 配对 CSP → fbcsp_permutation.py
```

> **重要：** 阶段 1 和阶段 2 当前由 `EEG_preprocessing.py` 的一次 `main()`
> 调用依次完成，不是两个可以分别执行的入口脚本。阶段 3 和阶段 4 必须在
> 阶段 2、阶段 3 正式输出成功后按顺序执行。

---

## 1. 项目目标和设计原则

### 1.1 目标

- 从原始 HDF5 读取连续 EEG、采样率、通道名、异步事件和标签映射。
- 在不删除原始时间轴的情况下，把幅值异常记录为 `bad_amplitude` 注释。
- 将多个短记录安全拼接为一个连续 Raw，并在接缝处保留 `bad_boundary` 注释。
- 使用标准电极位置完成坏通道插值，再执行即时平均参考。
- 通过人工查看 ICA 头皮图、源时序和成分属性，决定实际排除哪些成分。
- 在 ICA 正式发布后，再进行 8–30 Hz 滤波和任务 Epoch 质量控制。
- 同时发布一个保留完整审核窗口的 HDF5 和一个兼容 `ws.py` 的分类 HDF5。

### 1.2 关键原则

1. **不自动选择 ICA 伪迹成分。** 程序可以展示候选成分，但不会根据规则自动
   排除成分；`ICA_EXCLUDE_COMPONENTS` 必须由人工根据图形检查填写。
2. **阶段之间使用正式文件和校验凭据交接。** 阶段 4 不会在找不到 clean Raw
   时回退读取阶段 2 的 Raw。
3. **坏段通过注释传播。** 阶段 1/2 不直接裁剪连续数据；后续滤波和 Epoch
   拒绝根据 `bad_*`、`bad_boundary` 等注释处理。
4. **输出先写临时文件、回读验证，再替换正式文件。** 发生失败时尽量保留上一次
   成功输出；不要并发运行同一阶段或在发布过程中读取输出文件。
5. **时间单位和电压单位明确区分。** MNE 内部 EEG 单位是伏（V），最终 HDF5
   数据单位是微伏（µV）。

---

## 2. 目录结构

```text
eeg-preprocessing-pipeline/
├── config.example.py                 # 可公开的完整配置模板
├── config.py                         # 本地配置，由模板复制后填写，不公开个人路径
├── requirements.txt                  # 已验证环境的直接依赖及版本
├── EEG_preprocessing.py              # 阶段 1 + 阶段 2
├── eeg_reference.py                  # 坏通道检测、插值和平均参考辅助逻辑
├── ICA_processing.py                 # 阶段 3：ICA 和人工审核
├── EEG_processing.py                 # 阶段 4：滤波、Epoch 和 HDF5 发布
├── ws.py                             # clean HDF5 校验及 CSP + SVM 分类函数
├── main.py                           # 单频带 CSP 分类入口（会覆盖其既有结果）
├── permutation.py                    # 单频带 CSP 置换检验入口
├── fbcsp_stage4.py                    # 多频带阶段 4 与同源宽带 baseline 发布
├── fbcsp.py                           # FBCSP 变换器、grouped 搜索及嵌套 CV
├── fbcsp_main.py                      # FBCSP 与配对 CSP 分类入口
├── fbcsp_permutation.py               # FBCSP 专用置换检验和断点续跑
├── FBCSP_GUIDE.md                     # 方法、协议、运行与测试说明
├── processed_data_concatenation.py   # 已处理 HDF5 沿 Epoch 维拼接的独立工具
├── cache/                            # FIF、HDF5、ICA 模型和清单
├── reports/                          # ICA 审核报告和阶段 4 报告
├── CSP_result/                       # 本地单频带实验结果，不上传
├── FBCSP_result/                     # 本地多频带分类和置换结果，不上传
├── *_test.py                         # 阶段回归测试
└── 其他工具脚本                      # HDF5、XML、映射和 EEG 查看工具
```

项目脚本通过 `config.PROJECT_ROOT` 计算项目根目录，不依赖当前终端目录；但
运行命令仍建议先切换到项目目录，以便日志和测试路径更直观。

---

## 3. 环境和配置文件

### 3.1 已验证环境

已验证的运行环境为：

- Python 3.9（验证时为 3.9.25）
- MNE 1.8.0
- Windows
- 项目的直接依赖及验证过的版本见 `requirements.txt`；除科学计算和分类库外，
  代码还直接使用 `joblib`、`threadpoolctl`。

安装 Python 3.9 后，在项目根目录打开 **Windows CMD**，创建独立环境并安装依赖：

```bat
python --version
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip check
```

请确认第一行显示 Python 3.9，再创建环境；如系统有多个 Python 版本，
应先选择 Python 3.9 解释器。以后重新打开终端，先执行
`.venv\Scripts\activate.bat`；其他操作系统或 Conda 用户也可在对应环境中执行
`python -m pip install -r requirements.txt`。下文统一使用当前环境的 `python`。
这是已验证的直接依赖版本清单，而不是整个本机 Conda 环境的锁文件；
间接依赖由 pip 解析安装，不保证跨操作系统的二进制环境完全一致。
安装依赖也不会提供原始 EEG 数据，完整流水线仍需自备符合格式的输入文件。

### 3.2 从模板创建本地配置

首次使用且项目根目录下没有 `config.py` 时，在 Windows CMD 执行：

```bat
if not exist config.py copy config.example.py config.py
```

其他终端也可以通过文件管理器复制并重命名。已有 `config.py` 时保留原文件，
不要覆盖。程序只读取 `config.py`；只修改 `config.example.py` 不会改变运行配置。

模板使用项目根目录下的 `data/` 作为示例输入目录，编号范围为 1～10。
请在本地 `config.py` 中填写实际的 `DATASET_DIR`、文件模板、条件和输入编号；
示例数据并未随模板提供，实际处理前仍会检查文件是否存在。

模板的 `ICA_RUN_MODE="fit"`、`ICA_APPLY_CONFIRM=False`，
`ICA_EXCLUDE_COMPONENTS=()`。空列表不代表已经审核或保证数据无伪迹，
必须根据自己的模型重新 review、preview 并人工确认后才能 apply。
滤波、阈值、通道映射及时间窗保留为示例参数，不是适用于所有数据的推荐结论。
模板保留原有输出覆盖开关，重跑前请检查 `*_OVERWRITE` 并备份重要结果。

本地原始数据以及 `cache/`、`reports/`、`CSP_result/` 中生成的数据和报告
应独立保存，不作为此公开配置和文档的一部分。运行中产生的报告仍可能记录本机路径。

`config.py` 只定义配置，不会在导入时运行处理流程。
运行任何阶段前，先检查以下内容：

辅助工具也从本地 `config.py` 读取输入：采集信息和 XML 查看器使用
`ACQUISITION_VIEW_INPUT_PATH`，HDF5 结构/压缩查看器使用 `HDF5_VIEW_INPUT_PATH`。
MAT 单文件转换使用 `MAT_INPUT_PATH` 和 `MAT_OUTPUT_PATH`；批量转换使用
`MAT_DATASET_DIR`、`MAT_BATCH_START_SUB`、`MAT_BATCH_END_SUB`、`MAT_BATCH_N_SESSIONS`。
`convert_mat_to_clean_hdf5_test.py` 是历史命名的批量转换工具，不是单元测试。
转换工具针对注释中描述的 MAT 结构；私有检查报告不随源码发布。

| 配置组 | 主要配置项 | 作用 |
| --- | --- | --- |
| 原始输入 | `DATASET_DIR`、`INPUT_FILE_TEMPLATE`、`INPUT_CONDITION` | 生成原始 HDF5 路径 |
| 输入选择 | `INPUT_INDICES` 或 `INPUT_INDEX_START/END`、`EXCLUDED_INPUT_INDICES` | 决定文件和拼接顺序 |
| 元数据 | `SOURCE_DATA_UNIT`、`DEFAULT_SFREQ`、`ALLOW_METADATA_FALLBACK` | 控制单位、采样率和元数据回退 |
| 阶段 1 | `HIGH_PASS_*`、幅值阈值、`PREICA_BAD_AMPLITUDE_LABEL` | 高通和保护性坏段初筛 |
| 阶段 2 | `CONCATENATED_RAW_PATH`、覆盖开关、参考相关配置 | 长 Raw 发布 |
| 阶段 3 | `ICA_*` | ICA 模型、人工审核、预览和正式发布 |
| 阶段 4 | `STAGE4_*` | 连续带通、事件分段、HDF5 和报告 |

默认输入选择规则为：

1. `INPUT_INDICES` 非空时，按该元组顺序读取；
2. 否则读取闭区间 `INPUT_INDEX_START` 到 `INPUT_INDEX_END`；
3. 最后排除 `EXCLUDED_INPUT_INDICES`；
4. 选择结果不能为空，也不能含重复编号。

默认文件模板为：

```text
data/{index}-10s-{condition}.hdf5
```

其中 `{condition}` 只用于选文件，不直接决定最终的 `left/right` 标签。

### 3.3 配置变更后的重跑规则

- 原始输入、元数据、阶段 1 或阶段 2 参数变化：重新运行阶段 1/2，并重新拟合
  阶段 3 ICA。
- 阶段 3 输入、ICA 参数或排除列表变化：重新执行 `fit`（如有需要）、`review`、
  `preview` 和 `apply`。
- 只修改阶段 4 参数：重新运行阶段 4；但当前实现对导出规格进行了固定校验，
  不支持仅通过改配置任意改变输出格式。
- 不要把旧 ICA 成分编号直接套到新模型上。模型 ID、输入 SHA-256 和参数清单
  不匹配时，程序会拒绝复用旧模型或旧预览。

---

## 4. 阶段 1：原始 HDF5 预处理和坏段标记

### 4.1 入口和运行命令

入口文件：`EEG_preprocessing.py`

```bat
:: 请先在项目根目录打开终端
python EEG_preprocessing.py
```

该入口会先完成阶段 1，再立即进入阶段 2。阶段 1 的短 Raw 列表主要保留在本次
进程内，独立阶段 3 不读取“阶段 1 文件”，而是读取阶段 2 正式 FIF。

### 4.2 输入数据

每个原始 HDF5 通常需要包含以下信息：

- EEG 样本矩阵 `RawData/Samples`；
- 异步事件时间 `AsynchronData/Time`；
- 事件类型 `AsynchronData/TypeID`；
- 有效性或事件值 `AsynchronData/Value`；
- 采集 XML 元数据，包括采样率和通道名；
- `AsynchronSignalTypes` 对应的事件类型映射。

程序按配置顺序逐个读取文件，并将原始数据从 `microvolts` 转换为 MNE 使用的
`volts`。默认只保留 `Value == ACTIVE_EVENT_VALUE` 的事件。

### 4.3 处理过程

1. 解析采样率和通道名称，并检查通道名非空、唯一且数量正确。
2. 保留并校验采集元数据中的通道名称；设备通道映射和 `standard_1020` 坐标在
   阶段 2 拼接完成后统一应用。
3. 将有效采集事件映射为任务注释，并检查事件是否落在 EEG 样本范围内。
4. 对每个短 Raw 做 1 Hz 高通。默认使用四阶 Butterworth IIR；阶段 1 不做低通、
   不做平均参考、不做 ICA 和正式任务分段。
5. 用 `annotate_amplitude` 检查相邻采样点的突变，并额外执行绝对幅值阈值检查。
   两类异常区间合并后统一命名为 `bad_preica_amplitude`，保护后续坏通道分析和 ICA 拟合。
6. 保持连续数据和原始时间轴，不直接删除坏样本；坏区间通过 Annotation 传给
   后续阶段。
7. 如果 `RUN_EPOCH_REJECTION_CHECK=True`，用 `QC_EPOCH_TMIN=-1`、
   `QC_EPOCH_TMAX=4` 和指定 baseline 做试切，仅打印坏段能拒绝多少 Epoch，
   不会裁剪正式 Raw，也不会替代阶段 4 的最终审核。

### 4.4 阶段 1 输出和 QC 结果

阶段 1 的主要输出是当前进程中的短 Raw 列表。每个 Raw 包含：

- EEG 数据（MNE 内部单位为 V）；
- 任务事件 Annotation；
- `bad_preica_amplitude` 初筛坏段 Annotation；
- 原始通道顺序和采样率；电极标准化、坏通道修复和参考尚未执行；
- 尚未应用的新参考或投影。

终端 QC 输出用于回答“当前坏段标记是否能拒绝被污染的检查 Epoch”，但它不是
最终 Epoch 报告。最终保留数量和拒绝原因以阶段 4 的
`reports/stage4_epoch_report.json` 为准。

### 4.5 常见失败原因

- 输入文件不存在、编号选择为空或编号重复；
- HDF5 数据集缺失、形状不正确或样本数与事件数组不一致；
- XML 无法解析、缺少 `SampleRate`、通道数不匹配或通道名重复；
- 采样率、通道名、通道顺序、通道类型、单位或电极信息在不同文件间不一致；
- 数据含 NaN/Inf，或事件超出 EEG 时间范围。超范围事件默认跳过并警告，
  `SKIP_OUT_OF_RANGE_EVENTS=False` 时会直接报错。

---

## 5. 阶段 2：Raw 拼接、坏通道修复和平均参考

### 5.1 处理过程

阶段 2 紧接阶段 1 执行，主要步骤如下：

1. 检查短 Raw 的元数据兼容性，不静默修正采样率、通道顺序或其他冲突。
2. 使用 `mne.concatenate_raws()` 拼接短 Raw。MNE 自动处理注释时间偏移。
3. 将拼接产生的边界注释统一为 `bad_boundary`，并删除重复的 `EDGE boundary`。
   接缝处使用零时长边界注释，目的是阻止 Epoch 跨越两个独立试段。
4. 将完整历史设备通道名映射为标准 EEG 通道名，并设置 `standard_1020` 电极
   坐标；未知或混合命名会报错，不猜测电极位置。
5. 检测持续性坏通道。坏通道判定和修复由 `eeg_reference.py` 完成；满足条件的
   通道使用标准电极位置做球面样条插值，不把坏通道永久从数据中删除。
6. 对修复后的连续 Raw 应用即时平均参考，记录有效数据秩；阶段 3 ICA 将使用
   这个已经完成平均参考的连续数据。
7. 将长 Raw 写入同一目标目录的临时目录，回读检查数据、通道、采样率和注释，
   通过后原子替换正式输出。

### 5.2 正式输出

默认输出：

```text
cache/stage2_concatenated_raw.fif
```

该文件是阶段 3 的唯一 EEG 输入。它是连续 Raw，不是已经分好段的 Epoch 数据。
文件中保留任务注释、`bad_preica_amplitude` 和 `bad_boundary`，因此后续阶段可以识别
坏区间和拼接接缝。FIF 的 `info["description"]` 还会嵌入阶段 2 生成时间、输入
来源，以及参考方式、插值通道、有效秩和 montage 等紧凑摘要；完整检测记录由
函数返回，但当前阶段不单独发布 JSON 报告。

当前流程只接受单文件 FIF。若 MNE 保存产生分卷 FIF，程序会拒绝发布，并保留
上一次成功的正式输出。

### 5.3 阶段 2 的验证要求

正式发布前后会检查：

- 数据长度、通道名和采样率一致；
- 注释数量和描述一致；
- EEG 数据为有限实数，单位为 V；
- 不存在不允许的旧 boundary 注释；
- 输出文件可以被重新读取。

阶段 2 成功后，如果当前 Python 进程中已经加载过旧 Raw 或旧 ICA 对象，必须
重新加载 `CONCATENATED_RAW_PATH`；内存中的对象不会随磁盘文件自动更新。

---

## 6. 阶段 3：ICA 拟合、人工审核、预览和正式发布

### 6.1 入口和基本参数

入口文件：`ICA_processing.py`

阶段 3 默认使用：

- 阶段 2 长 Raw：`config.ICA_INPUT_RAW_PATH`；
- 32 个设备 EEG 通道和 `standard_1020` 电极坐标；
- FastICA；
- `ICA_N_COMPONENTS` 个成分（默认配置为 15）；
- 随机种子 42；
- 至少 1 Hz 高通后的数据；
- 拟合时跳过 `bad*` 注释覆盖的样本。

ICA 成分数是分解分析参数，不是输出通道数。即使拟合 15 个成分，正式 clean
Raw 仍保留全部 32 个 EEG 通道；重建也保留 PCA 残余信息。

### 6.2 推荐操作顺序

严格按照以下顺序执行：

```text
fit → review（或 inspect）→ 修改 ICA_EXCLUDE_COMPONENTS → preview → apply
```

命令行方式：

```bat
python ICA_processing.py fit
python ICA_processing.py review
python ICA_processing.py preview
python ICA_processing.py apply --confirm
```

也可以使用 VS Code 的“运行 Python 文件”按钮：

1. 选择正确的 Python 解释器；
2. 在 `config.py` 设置 `ICA_RUN_MODE`；
3. 运行 `ICA_processing.py`，不要运行 `config.py`；
4. `ICA_RUN_MODE` 可取 `fit`、`review`、`inspect`、`preview`、`apply`。

无参数启动时使用配置文件；显式提供命令行参数时使用命令行模式，不会自动串联
多个阶段，也不会自动确认发布。

### 6.3 `fit`：拟合或复用 ICA 模型

```bat
python ICA_processing.py fit
python ICA_processing.py fit --force
```

`fit` 的行为：

- 若输入文件和拟合参数都匹配，默认复用已有模型；
- `--force` 强制重新拟合；
- 使用未被 `bad*` 注释覆盖的样本；零时长 `bad_boundary` 会在拟合副本中按
  `ICA_BOUNDARY_PADDING_SECONDS` 向前后扩展，以避免 ICA 学到跨接缝结构；
- 只拟合和保存模型，不减去 ICA 成分，也不生成 clean EEG；
- 未收敛模型不会作为正式模型发布。

默认模型和清单：

```text
cache/stage3-ica.fif
cache/stage3_ica_manifest.json
```

清单记录模型 ID、拟合时间、来源文件绝对路径和 SHA-256、通道顺序、采样率、
拟合参数、成分数、拟合样本数、迭代次数、收敛状态和被省略的时间。输入或参数
变化后，旧清单会被识别为不匹配。

### 6.4 `review` / `inspect`：人工审核

```bat
python ICA_processing.py review
```

审核窗口通常包括：

1. 分页显示 ICA 头皮图；
2. 浏览 ICA 源时序；
3. 查看指定成分的属性图和功率谱；
4. 关闭图形窗口后进入下一页或下一项。

`ICA_REVIEW_COMPONENTS` 或命令行 `--components` 只决定哪些成分显示属性图，
不是排除建议。源浏览器中点击成分可以得到候选编号，但候选不会自动写入正式
排除列表，也不会改变 ICA 模型。

属性图中的 2 秒窗口只用于可视化，不是任务 ERP，也不是阶段 4 的正式分段窗口。
默认审核图片和候选记录写入：

```text
reports/ica_review/<model_id>/
├── ica_components_*.png       # 分页头皮图
├── ICAxxx_properties.png      # 指定成分属性图
└── candidate_components.json   # 人工查看得到的候选编号，仅供参考
```

设置 `ICA_SHOW_PLOTS=False` 时只导出图片，不弹出交互窗口。

### 6.5 `preview`：检查拟议排除效果

先在 `config.py` 中人工填写 0-based 成分编号，例如：

```python
ICA_EXCLUDE_COMPONENTS = (0, 4)
```

编号只是示例，不能据此推断成分 0 或 4 一定是伪迹。然后运行：

```bat
python ICA_processing.py preview
```

`preview` 会在内存副本中排除配置的成分，并生成：

```text
reports/ica_review/<model_id>/
├── ica_overlay.png             # ICA 排除前后的叠加对比
├── before_after.png            # 指定通道的前后波形对比
└── preview.json                # 本次输入、模型和排除列表的匹配凭据
```

预览不会发布 `stage3_ica_clean_raw.fif`。如果模型、输入或排除列表发生任何变化，
必须重新运行 `preview`；旧凭据不能用于新的选择。

### 6.6 `apply`：正式发布 clean Raw

确认预览效果后，可任选一种发布方式。

VS Code 无参数运行方式：在 `config.py` 中设置：

```python
ICA_RUN_MODE = "apply"
ICA_APPLY_CONFIRM = True
```

然后运行 `ICA_processing.py`。命令行方式不读取该确认开关，直接使用：

```bat
python ICA_processing.py apply --confirm
```

正式发布前会再次验证输入哈希、模型哈希、预览凭据和排除列表。通过后输出：

```text
cache/stage3_ica_clean_raw.fif
reports/ica_review/<model_id>/applied_components.json
```

clean Raw 的特点：

- 保留全长连续时间轴和全部 EEG 通道；
- 用 ICA 后重新检测出的 `bad_amplitude` 替换初筛幅值注释；保留 `bad_boundary` 等永久标记；
- 不在阶段 3 进行任务 Epoch 分段；
- 文件的 `info["description"]` 嵌入来源、模型、排除列表和发布时间信息。

`applied_components.json` 是阶段 4 验证 clean Raw 来源的正式凭据，包含模型 ID、
输入来源哈希、排除成分、应用时间、输入路径和 clean FIF 的 SHA-256。发布完成后
若使用 VS Code 配置方式，发布完成后应立即将 `ICA_APPLY_CONFIRM` 改回 `False`，
避免误操作再次发布。

### 6.6 ICA 前初筛与 ICA 后复检

阶段 1 的 `bad_preica_amplitude` 只做保护性初筛。阶段 3 的 preview 和 apply
在重建后的连续信号上重新执行突变与绝对幅值检测（在阶段 4 最终带通之前），
精确删除 `bad_preica_amplitude` 和旧版阶段 1 的 `bad_amplitude`，再生成最终
`bad_amplitude`。没有超限的旧区间不再仅因初筛标记被阶段 4 拒绝；仍然异常或
新出现的异常区间继续拒绝。预览的上下两幅图分别显示各自的注释。

独立配置项：`POSTICA_AMPLITUDE_PEAK_UV`（默认 100 µV）、
`POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV`（默认 300 µV）、两者对应的
`*_MIN_DURATION`（默认 0.004 秒）及 `POSTICA_AMPLITUDE_MERGE_GAP_SECONDS`（默认 0）。
这些默认值沿用初筛量级，仍需结合设备和实验数据审核；修改它们必须重新 preview/apply，
但仅修改复检阈值不需要重 fit。阶段 4 会拒绝复检参数不匹配或没有复检凭据的旧 clean。

其他 `bad*` / `edge*` 标记一律保留。例如 `bad_boundary`、`bad_acq_skip`、
`bad_dropout`、`bad_clipping`、`bad_flat` 不会因 ICA 后幅值降低而被撤销。
硬件或数据损坏必须使用独立注释名，不能混用两种可复检幅值名称。
本次改动不新增丢包、削顶、局部平线的自动识别算法；只保证已有永久标记不被删除。

`amplitude_recheck` 同时写入发布凭据、FIF 内嵌记录及阶段 4 报告，包含复检参数、
ICA 前/后幅值区间、永久标记的数量和合并后总时长。前后坏段数之差不是“挽救的 Epoch 数”。
阶段 4 另按 MNE 实际返回的拒绝原因统计 `pre_ica`、`post_ica`、`permanent`；
MNE 不保证列出全部重叠坏段，因此这些计数不是完整重叠区间统计。

已有阶段 2 输入及 ICA 模型若来源哈希仍匹配，可直接重新 preview、人工确认后 apply，
再运行阶段 4，无须为改注释名称重跑阶段 1/2。若重跑阶段 1/2 导致输入变化，
则必须重新 fit/review/preview/apply。旧版 `bad_amplitude` 按历史幅值检测标记兼容处理。

---

## 7. 阶段 4：滤波、事件分段、坏 Epoch 拒绝和双 HDF5 发布

### 7.1 入口和运行命令

入口文件：`EEG_processing.py`

```bat
:: 请先在项目根目录打开终端
python EEG_processing.py
```

阶段 4 只接受正式的：

```text
cache/stage3_ica_clean_raw.fif
```

它不会在缺少 clean Raw 时回退读取阶段 2 FIF，也不会重新拟合 ICA、重新插值或
重新应用平均参考。

### 7.2 处理过程

1. 校验阶段 3 clean Raw、`applied_components.json`、模型清单和各 SHA-256 的来源
   一致性。
2. 检查输入为相对时间 Raw：`first_samp=0`，无绝对测量日期，32 个 EEG 通道，
   采样率满足配置滤波的 Nyquist 条件，有标准电极坐标，并且已经完成平均参考和至少 1 Hz 高通。
3. 在连续 Raw 副本上执行 8–30 Hz FIR `firwin` 零相位带通。滤波显式跳过
   所有 `bad*` / `edge*`（含永久坏段及 ICA 后幅值标记），避免跨接缝或坏段
   传播滤波响应。
4. 根据 `STAGE4_EVENT_ID` 创建完整审核 Epoch，时间窗和基线分别来自
   `STAGE4_EPOCH_TMIN/TMAX` 和 `STAGE4_BASELINE`。
5. 使用 `reject_by_annotation=True` 拒绝与坏注释相交或超出连续数据范围的 Epoch。
   `edge*` 在分段副本上增加 `bad_permanent_edge` 别名，确保也参与拒绝。
6. 从审核通过的 Epoch 按 `STAGE4_WS_TMIN/TMAX` 裁出分类窗口；采样点数动态计算。
7. 按 `STAGE4_OUTPUT_UNIT/DTYPE` 转换 EEG 单位和浮点类型，标签为 `int32`；两个 HDF5、完整
   报告都回读/校验后才替换正式文件。

### 7.3 事件和标签

阶段 4 默认使用以下两个任务注释；名称和整数标签可在 `STAGE4_EVENT_ID` 中修改：

| 注释 | HDF5 标签 |
| --- | ---: |
| `left` | 1 |
| `right` | 2 |

`bad_*` 和 `edge` 类注释用于质量控制，不会作为分类标签导出。原采集 HDF5 中
的 `TypeID` 数字不需要与阶段 4 的导出标签相同；阶段 4 使用
`STAGE4_EVENT_ID` 定义的注释名称到整数标签映射。

---

## 8. 阶段 4 输出数据、报告和审核结果

### 8.1 输出文件清单

| 文件 | 内容和用途 |
| --- | --- |
| `cache/stage4_epochs_full.hdf5` | 通过配置的完整窗口坏段/越界审核后保留的 Epoch；保留完整窗口供质量检查和复核 |
| `cache/stage4_epochs_ws.hdf5` | 从完整审核通过的 Epoch 中按配置裁出的分类输入 |
| `reports/stage4_epoch_report.json` | 阶段 4 的拒绝统计、标签数量、来源信息和两个 HDF5 哈希 |
| `reports/ica_review/<model_id>/applied_components.json` | 阶段 3 的正式 ICA 排除凭据，是阶段 4 的来源证明 |

### 8.2 两份 HDF5 的结构

两份阶段 4 HDF5 都包含以下数据集：

| 数据集 | 形状/类型 | 说明 |
| --- | --- | --- |
| `clean_eeg` | `(n_epochs, 32, n_times)`，配置的浮点类型 | EEG 数据，单位由 `STAGE4_OUTPUT_UNIT` 指定 |
| `labels` | `(n_epochs,)`，`int32` | 类别名称和整数值由 `STAGE4_EVENT_ID` 指定 |
| `channel_names` | `(32,)` | 通道名，顺序与 EEG 第 1 维一致 |
| `times` | `(n_times,)` | 相对事件时间，单位秒 |
| `event_samples` | `(n_epochs,)` | 每个保留 Epoch 对应的事件采样位置 |

两份文件的主要区别：

| 文件 | 窗口 | 时间点数 |
| --- | --- | ---: |
| `stage4_epochs_full.hdf5` | `STAGE4_EPOCH_TMIN`～`STAGE4_EPOCH_TMAX` | 按输入采样率自动确定 |
| `stage4_epochs_ws.hdf5` | `STAGE4_WS_TMIN`～`STAGE4_WS_TMAX` | 按输入采样率自动确定 |

文件属性还记录采样率、单位、窗口、滤波规格、来源和导出元数据。`ws.py` 至少
要求 `clean_eeg`、`labels` 和 `sampling_rate`，阶段 4 发布的 HDF5 同时提供
`channel_names`、`times` 和 `event_samples`，可用于追溯。

### 8.3 JSON 审核报告

阶段 4 还会在终端打印候选 Epoch 总数、丢弃总数、因 `bad*/edge` 坏段注释丢弃的去重
Epoch 数、最终保留数和各项拒绝原因。分类窗口只裁剪已保留 Epoch 的时间范围，不会
额外丢弃试次。一个 Epoch 可能同时命中多个拒绝原因，因此各原因计数之和不一定等于
丢弃 Epoch 总数。

`stage4_epoch_report.json` 用于审计本次发布，重点包括：

- 原始任务事件数量和 `left/right` 类别数量；
- 保留 Epoch 数量和拒绝 Epoch 数量；
- 按坏注释、边界或越界等原因统计的拒绝结果；
- 阶段 3 clean Raw、ICA 模型和 apply 凭据的来源信息；
- 完整 HDF5 的 `full_sha256`；
- ws HDF5 的 `ws_sha256`；
- 导出窗口、采样率、通道数、时间点数和生成时间。

报告只有在两个 HDF5 写入并验证成功后才作为最后的提交记录发布。它不是跨多
文件系统事务：若进程在替换文件期间被强制终止，应以报告中的 SHA-256 重新检查
文件，必要时重新运行阶段 4。

### 8.4 “最终结果”的判定

一次完整处理只有同时满足以下条件，才视为成功：

1. `stage3_ica_clean_raw.fif` 存在且与 `applied_components.json` 哈希匹配；
2. `stage4_epochs_full.hdf5` 和 `stage4_epochs_ws.hdf5` 均可读取；
3. 两份 HDF5 的数据集、dtype、形状、标签和属性通过回读验证；
4. `stage4_epoch_report.json` 存在，且其中两个 SHA-256 与实际文件一致；
5. `stage4_epochs_ws.hdf5` 能通过 `ws.validate_clean_hdf5()`。

建议在发布后使用以下检查：

```bat
python -c "import config; import ws; ws.validate_clean_hdf5(config.STAGE4_WS_OUTPUT_PATH); print('stage4 ws HDF5 OK')"
```

---

## 9. 使用 `ws.py` 进行分类

`ws.py` 提供 clean HDF5 格式验证、读取和 CSP + SVM 分类函数。它读取的数据格式
约定为：

```text
clean_eeg: (n_epochs, n_channels, n_times)
labels:    (n_epochs,)
sampling_rate: HDF5 属性
```

读取时 EEG 会转为 `float64` 供 MNE CSP 使用；默认分类流程为**外层 10 折、内层
5 折的嵌套分层交叉验证**，两层均保持 `shuffle=False`，使用准确率评分。
每个外层训练折独立运行 `GridSearchCV`，搜索对象为完整的
`Pipeline(CSP → StandardScaler → 线性 SVM)`，而非预先提取的 CSP 特征。
每个候选/内层折都重新拟合 CSP、标准化器和 SVM；内层验证折只参与评分。
选定参数后在完整外层训练折重新拟合 Pipeline，最后仅在外层测试折评估。
外层测试数据不参与参数选择，返回的平均准确率是 10 个外层测试分数的算术平均。

默认网格在 `ws.py` 的 `DEFAULT_PARAM_GRID` 中：

| 参数 | 候选值 |
| --- | --- |
| `csp__reg` | `0.1, "ledoit_wolf"` |
| `csp__n_components` | `2, 4` |
| `csp__log` | `True` |
| `svm__C` | `0.1, 1.0, 10.0` |

固定 `csp__norm_trace=False`、SVM `kernel="linear"`。MNE 的 `log=False` 分支
使用训练数据的统计量标准化平均功率，并非单纯输出原始功率；两分支后均保留
Pipeline 中的 `StandardScaler`。32 通道时共 12 个候选，默认 reference 执行 600 次内层
Pipeline 拟合和 10 次外层训练集重拟合，计算量明显高于单层交叉验证。
默认 `n_jobs=1`；底层数学库线程默认限制为每进程2个，MNE只输出WARNING及以上日志。
可显式设为2或4个进程，但需要更多内存，不建议直接使用 `-1`。
`train_test` / `nested_cv` 支持 `cache_transformers=True`：只在相同训练子集、标签和
CSP参数下复用变换，每个外层折使用独立临时缓存并及时删除，不跨置换共享。
分类函数默认不缓存以兼容原有调用；单频带置换仅在选择 reference 引擎时默认开启缓存，
grouped 引擎不使用磁盘缓存。FBCSP 的缓存与搜索规则见第 9.2 节。

默认成分候选会排除超过通道数的值；自定义网格中的非法成分数直接报错。
不使用完整数据估计有效秩来调整网格。所有外层训练折的每类样本数都必须满足
内层折数要求，样本不足不自动降低折数。拟合错误直接抛出，不静默跳过候选。

可以在 Python 中直接调用：

```python
from ws import train_test

average_score, fold_scores = train_test(
    r"cache\stage4_epochs_ws.hdf5",
    k_fold=10,
    inner_k_fold=5,
)
print(average_score)
print(fold_scores)
```

需要审核选参过程时，传 `return_details=True`，接收第三个返回值 `details`；其中
包含实际网格、各外层折训练/测试索引（相对于输入 HDF5 epoch 顺序）、最优参数、
内层最佳分数和外层测试分数。默认仍返回两个值，兼容原有调用。
可传 `param_grid` 字典覆盖默认网格；省略的维度使用 Pipeline 固定默认值
（成分数 2、reg="oas"、log=False、C=1.0），而非继续搜索该维度。

运行正式入口：

```bat
python main.py
```

该入口默认进行 10×5 嵌套交叉验证，并写入原有分类结果路径（会覆盖旧结果）。
TXT 逐折记录最优参数及内外层分数；MAT 保留 `accuracy`、`fold_scores`、`config`，
新增 `best_inner_scores` 和 `nested_cv_details_json`（JSON 保留混合类型参数和划分索引）。
各外层折可以选择不同的最优参数，不能用外层测试成绩再次挑选“最优折”来报告。

边界：本次嵌套验证覆盖 CSP/标准化/SVM 及其超参数选择，不会将阶段 1–4 的
ICA、人工审核等预处理移入折内，因此不代表整个原始 EEG 处理链均经过折内拟合。
分层划分也不等同于按被试、session 或采集时段分组的独立泛化评估。

注意：当前 `ws.py` 主要提供分类函数；它本身不负责阶段 1–4 的处理，也不会
自动检查完整审核报告与 ICA 凭据。正式实验应先完成阶段 4，并保存对应报告，
再将 `stage4_epochs_ws.hdf5` 作为分类输入。

---

### 9.1 Epoch 级标签置换检验

入口为 `permutation.py`（`permutation_test.py` 现在是合成数据测试，不是实验入口）。默认复用
`CSP_result\classification_result\classification_result.mat`
中的原始准确率（完整精度），**不会重新训练原始标签，也不覆盖分类结果**。
必须先完成一次正式分类。先做只读校验（不拟合、不生成输出）：

```bat
python permutation.py --validate-only
```

正式开始 100 次置换的命令如下；计算量很大，不会在测试期间自动启动：

```bat
python permutation.py --n-permutations 100 --seed 42
```

`--n-permutations` 默认100，`--seed` 默认42，`--n-jobs` 默认1（只并行内层网格搜索，
置换之间和外层折之间均串行）。`--native-threads` 默认2，限制每个进程的底层线程；
内层使用 loky 进程后端并限制预分发为 `n_jobs`，避免多层并行与过量排队。
置换入口默认 `--search-engine grouped`，按内层折和 reg 分组搜索，不使用磁盘缓存。
`--search-engine reference` 保留原 GridSearchCV；仅此引擎默认启用 Pipeline 磁盘缓存，
可加 `--no-cache` 关闭作对照。`train_test` / `nested_cv` 为兼容既有调用仍默认 reference，
可显式传 `search_engine="grouped"`。
可用 `--observed-mat`、`--hdf5`、`--stage4-report` 显式指定绝对路径。
`--output-dir` 必须是尚不存在的新目录，避免覆盖旧实验；默认在
`CSP_result\permutation` 下创建 UTC 时间戳子目录。

**统计过程：**每次从原始 `labels` 取独立随机排列，单位为 epoch，保持类别总数；
不更改 EEG 的 epoch 顺序、通道或时间点，也不修改 HDF5。允许随机排列中出现原位置
未变的 epoch、重复的标签排列或原始排列，不强制重抽。这符合有放回 Monte Carlo 抽样。
输入 EEG 不变不代表 CSP 特征不变：CSP 依赖标签，必须重新拟合。

每次使用 MAT 记录的实际参数网格（当前默认为10×5、12组），而非只用原始最优参数；
根据本次置换标签重新生成分层折，保持 `shuffle=False` 的划分规则，而不是冻结
原始标签产生的具体索引；每次置换、每个训练折都会重新选择参数和拟合。
grouped：同一个内层训练子集、标签和 reg 只拟合一次空间滤波器，再按各成分数
计算训练/验证功率特征，分别做 log/非log 变换和训练集 StandardScaler，最后评估所有C。
不跨折或置换复用；验证标签只参与评分。n_components/log不影响当前固定MNE配置的
协方差和滤波器求解，因此无需为它们重复做特征分解。
为保留数值运算路径，2成分功率不是从4成分计算结果截取，而是用对应的滤波器重新投影；
非log分支保留 MNE 的训练均值/标准差归一化及后续 StandardScaler。
选择参数时严格维持 ParameterGrid 候选顺序、各内层折等权平均及并列首选规则。
外层重拟合仍使用原 MNE CSP→StandardScaler→SVC Pipeline。
统计量仍为外层测试准确率的等权平均，保存的实际网格全部保留。
当前默认每次置换有600个内部选参/评分组合及10次外层重拟合，grouped 的 CSP
拟合数为10×(5×2+1)=110，SVM拟合次数不变。reference 无缓存为610次；磁盘缓存
只复用完全相同变换器参数。grouped 去除 EEG 哈希、临时文件写读和按 C 重复的验证数据变换。
grouped 仅内层折并行，每个任务处理一个完整内层折，最多5个任务；外层及置换串行。
不支持当前受限网格以外的 CSP 参数；不能未经验证扩展到其他 component_order/cov_est。
每折打印耗时，每次置换记录耗时，并按本次进程已完成的置换估计剩余时间。

设原始准确率为 A，N 次置换准确率为 A_i，输出单侧经验 P 值：

```text
p = (1 + count(A_i >= A)) / (N + 1)
```

并列计入（仅用1e-12绝对容差处理浮点求和舍入），不使用简单的 count/N，P 不会为0。
N=100 时最小可分辨 P 为1/101≈0.0099；需要更精细估计时应事先指定更大 N，
不要根据中途结果决定停止或不断换种子挑选显著结果。

输出：
- `permutation_results.json`：训练前保存原始结果和配置，每完成一次置换就原子更新。
  包含输入/原始 MAT 哈希、种子、标签排列索引、每次标签、外层分数、各折最优参数、
  内层分数和划分索引。只有 `status="completed"` 才有最终 `p_value`。
- `permutation_summary.mat`：全部置换准确率、逐折分数、原始准确率、P 值与超越次数。
- `permutation_summary.txt`：逐次准确率及最终 P 值，便于直接阅读。

异常或用户中断会保留已完成记录并标记 `failed`，不计入失败的置换、不输出部分结果的
最终 P 值。支持显式 `--resume` 续跑（不是重新抽种子），例如：

```bat
python permutation.py --resume CSP_result\permutation\YOUR_RUN_DIRECTORY\permutation_results.json --n-permutations 100 --seed 42 --n-jobs 1 --native-threads 2
```

将 `YOUR_RUN_DIRECTORY` 替换为自己已有的运行子目录名；不要原样执行该示例。
可在同一命令增加 `--validate-only` 只读检查，不写入、不训练。
续跑严格核对输入/原始结果哈希、运行库版本、折数/网格、已保存置换标签序列和折索引、
逐折分数与均值一致性。N和seed必须与原检查点一致；原始任务必须已经停止。
若已完成 k 次，则从第 k+1 次完整重跑，不丢弃已完成结果，也不从中途外层折继续。
优化运行设置可改变，每次启动记录于 `execution_sessions`，耗时单独记录于
`permutation_timings`，不改变统计分数详情。已完成全部N次时只验证并返回。
续跑使用同名 `.lock` 防止两个新版续跑进程同时写入；若进程被强杀遗留锁，必须确认
没有活跃任务再手工移除。旧版进程不认识该锁，不能同时运行。
分组只减少重复计算，数学线程变化仍可能有微小浮点差异；不改变CV协议和网格。
运行期间不要重新发布输入 HDF5 或重新运行分类覆盖观测 MAT。

局部性能与等价验证工具：`profile_nested_search.py`。
仅运行真实 EEG 的第一次置换中指定的一个外层折（默认第1折），不更新正式结果，
报告每个候选每个内层折的分数、最优参数、外层预测、cProfile和Windows进程峰值工作集。
需显式指定 `--engine reference` 或 `--engine grouped` 和不存在的 `--output` 绝对路径。
请在自己的数据和环境中比较两个引擎的候选分数、预测和耗时。
带 profiler 的局部时间不能用于保证完整置换检验在固定时间内结束。

身份核验：新版分类结果直接保存输入 SHA256、CV 协议和运行库版本；旧结果缺少这些
字段时，必须通过阶段4报告的 run_id、输出路径和 SHA256 核验，且会警告无法完全证明
历史运行环境。新旧结果都核对数据元数据、网格、折数、外层索引、分数一致性。

**适用边界：**全局 epoch 置换假设零假设下标签在这些 epoch 间可交换。如果数据包含
不同被试/session/实验块或明显时间相关性，可能需要按实验设计限制的块内置换；
本实现不自动推断分组。检验针对固定的已预处理 EEG 和此分类流程，不检验上游 ICA、
人工筛选或事后反复选择时间窗/频带等步骤，也不自动校正多个实验的多重比较。

---

### 9.2 FBCSP-SVM 与配对 CSP（grouped-concat）

此分支不替代原三维 CSP 输入。`fbcsp_stage4.py` 从正式阶段 3 ICA clean Raw
分别生成 8–12、12–16、16–20、20–24、24–30 Hz 五个子带，以及同源 8–30 Hz
宽带 baseline；各分支直接滤波，不先宽带再子带。参考分支执行 Epoch QC，
各频带必须具有完全相同的标签、事件与试次顺序，否则拒绝发布。

已有本地 `config.py` 的用户，应从模板补齐 `FBCSP_BANDS`、四个
`FBCSP_*_PATH` 和 `FBCSP_OVERWRITE`，不要覆盖自己的路径或 ICA 审核配置。
默认输出为：

| 产物 | 项目根目录下的路径 |
| --- | --- |
| 完整审核窗口、四维多频带 EEG | `cache/stage4_fbcsp_epochs_full.hdf5` |
| FBCSP 分类输入 | `cache/stage4_fbcsp_epochs_ws.hdf5` |
| 同源宽带 CSP 分类输入 | `cache/stage4_fbcsp_baseline_ws.hdf5` |
| 发布报告及文件 SHA-256 | `reports/stage4_fbcsp_epoch_report.json` |

FBCSP 数据维度为 `(epochs, bands, channels, times)`。每带独立拟合 MNE CSP，
将各带功率特征按带序拼接，再接 `StandardScaler → 线性 SVM`；不含 mRMR、PCA
或自动频带选择。默认参数与上表相同，CSP 参数前缀改为 `fbcsp__`，其中
`n_components=2/4` 指**每个频带**的成分数。

**分组搜索与训练隔离：**每个内层训练折、每个 `reg`、每带只拟合一次 CSP，
显式保持 `cov_est="concat"`；每带按候选成分数截取空间滤波器，再生成特征。
每组 `(reg, n_components, log)` 独立提取特征并仅用训练特征拟合 Scaler，
只在不同 SVM C 之间复用同组特征。选参后在完整外层训练集重拟合整个 Pipeline。
这里的 grouped 指计算分组，**不是按被试/session 分组交叉验证**。
默认五带共需 `10 × (5 × 2 × 5 + 5) = 550` 次带级 CSP 拟合（含外层重拟合）。
不启用 Pipeline 磁盘缓存；`cache_transformers` 仅保留 API 兼容。
Python API 的 `search_engine="reference"` 可运行无缓存 GridSearchCV 作等价对照。

完成阶段 3 `apply` 后运行（无需先执行旧阶段 4）：

```bat
python fbcsp_stage4.py
python fbcsp_main.py --n-jobs 1 --native-threads 2
```

`FBCSP_OVERWRITE=False` 默认拒绝覆盖已有阶段 4 输出。**若输入、设置和已发布
文件未变，不必重跑阶段 3/4，直接重新分类即可。** 确需重新发布时，先审核输出路径
再显式允许覆盖；新输入不能继续使用旧观测分类或旧置换检查点。

分类默认在相同外层测试索引下同时评估 FBCSP 与同源宽带 CSP；可加
`--fbcsp-only` 跳过配对模型。默认在 `FBCSP_result/` 创建 UTC 时间戳目录，
也可用 `--output-dir` 指定尚不存在的新目录。保存：

- `classification_result.json`：完整观测结果，最后发布，作为完成标记及置换输入；
- `fbcsp_classification_result.mat`：准确率、各折分数和完整 JSON；
- `results.txt`：文本摘要。

JSON 包含各折索引、最优参数、预测、decision function、OOF 指标及配对准确率差。
`accuracy` 是各外层折准确率的等权平均；`metrics.oof_accuracy` 是汇总全部测试
预测后的准确率，折大小不等时两者可能不同。配对准确率差本身不代表统计显著。

#### FBCSP 专用置换：先校验，再启动

下例使用 CMD 环境变量；将 `OBSERVED` 改为实际新分类结果的 JSON 路径，
将 `PERM_OUTPUT` 改为尚不存在的新目录。不要预先创建输出目录。

```bat
set "OBSERVED=FBCSP_result\your_classification_run\classification_result.json"
set "PERM_OUTPUT=FBCSP_result\your_new_permutation_run"
python fbcsp_permutation.py --observed "%OBSERVED%" --validate-only
python fbcsp_permutation.py --observed "%OBSERVED%" --n-permutations 100 --seed 42 --output-dir "%PERM_OUTPUT%" --n-jobs 1 --native-threads 2
```

`--observed` 是必填参数；直接点击运行此脚本不会自动开始置换。
`--validate-only` 只读检查观测结果、发布文件哈希、网格、划分及运行库版本，不训练。
默认 `--n-permutations 100 --seed 42`，每次对 Epoch 标签全局置换，所有频带共享
同一标签向量，并根据置换标签重新分层，完整重跑嵌套选参及 CSP/Scaler/SVM。
固定无标签滤波不重跑，真实标签拟合的 CSP 不复用。

检查点为置换目录中的 `permutation_results.json`，每完成一次置换即保存。
中断后用**同一观测 JSON、总次数及 seed**续跑，不同时传 `--output-dir`：

```bat
python fbcsp_permutation.py --observed "%OBSERVED%" --resume "%PERM_OUTPUT%\permutation_results.json" --n-permutations 100 --seed 42 --n-jobs 1 --native-threads 2
```

检查点核验数据与观测哈希、协议、环境版本、网格和随机序列；残留 `.lock` 文件
只能在确认没有运行进程后处理。当前协议分别为：

- FBCSP：`nested_fbcsp_v2_grouped_concat`；
- 配对 CSP：`nested_csp_paired_v2_grouped_concat`；
- FBCSP 置换：`permutation_fbcsp_v2_grouped_concat`。

历史 72 候选分类与旧 FBCSP 置换检查点不能用于本实验，不得修改协议标识后复用。
单频带 `permutation.py` 的 MAT/检查点与此 JSON 入口也不能混用。

单侧 Monte Carlo 检验采用 `p = (1 + 随机准确率 ≥ 观测准确率的次数) / (N + 1)`，
包含并列值；100 次置换的最小可报告 p 值为 `1/101 ≈ 0.0099`，不是 0。
置换次数应事先确定，不按中途 p 值提前停止。检验只回答 FBCSP 相对随机标签
是否显著，**不检验 FBCSP 是否显著优于 CSP**。全局置换要求试次标签可交换；
存在 run/session 或时间依赖时应另行设计受限置换，当前入口不提供该模式。
本检验也不声称 ICA、人工审核等上游预处理在 CV 折内拟合。

---

## 10. 运行顺序（推荐）

### 10.1 完整正式流程

```bat
:: 请先在项目根目录打开终端

:: 阶段 1 + 阶段 2
python EEG_preprocessing.py

:: 阶段 3：拟合
python ICA_processing.py fit

:: 阶段 3：人工查看
python ICA_processing.py review

:: 修改 config.py 中的 ICA_EXCLUDE_COMPONENTS 后预览
python ICA_processing.py preview

:: CLI 通过 --confirm 明确确认并正式发布
python ICA_processing.py apply --confirm

:: 阶段 4
python EEG_processing.py
```

如果使用 VS Code 配置方式执行 `apply`，完成后要把 `ICA_APPLY_CONFIRM` 改回
`False`。每次只运行一个处理进程，不要在阶段 2/3/4 发布过程中打开或复制正式
输出文件。

### 10.2 VS Code 方式

阶段 1/2 和阶段 4 可以直接运行对应 Python 文件。阶段 3 通过修改
`ICA_RUN_MODE` 选择动作；一次点击只执行一个动作，不会自动从 `fit` 进入
`review`、`preview` 或 `apply`。

### 10.3 旧的独立脚本

目录中仍保留 `artifact_remove.py`、`eeg_viewer.py`、`original_clean_comparison.py`
等历史或辅助脚本。它们不属于上述四阶段正式发布链路，除非明确了解其输入格式和
输出契约，否则不要用它们替代 `EEG_preprocessing.py`、`ICA_processing.py` 或
`EEG_processing.py`。

---

## 11. 常见失败原因和恢复方式

| 现象 | 原因 | 恢复方式 |
| --- | --- | --- |
| 找不到原始 HDF5 | `DATASET_DIR`、模板或编号配置不正确 | 检查生成的绝对路径和 `VALIDATE_INPUT_FILES` 报错列表 |
| 元数据校验失败 | XML、采样率、通道名或通道顺序不一致 | 修正源文件/配置；不要用回退值掩盖真实通道问题 |
| 阶段 2 拒绝发布 | 元数据冲突、数据非法或生成分卷 FIF | 处理输入一致性；当前只支持单文件 FIF |
| ICA 模型无法复用 | 输入 SHA-256 或拟合参数变化 | 运行 `ICA_processing.py fit --force`，然后重新审核 |
| ICA 拟合不收敛 | 坏样本过多、秩不足或成分数过高 | 检查坏段和平均参考，降低 `ICA_N_COMPONENTS` 或修正数据后强制重拟合 |
| `apply` 提示需要 preview | 没有当前模型/当前排除列表的预览凭据 | 先运行 `preview`，确认图形效果后再 apply |
| 阶段 4 拒绝 clean Raw | 没有正式 apply 凭据、哈希失配或误用了阶段 2 FIF | 完成阶段 3 `apply`；不要手工复制/改名绕过校验 |
| 阶段 4 提示采样率/通道不符合 | 当前 ws 发布契约要求 32 通道、250 Hz 和标准电极位置 | 修正上游元数据或使用符合契约的输入 |
| Epoch 数量少于预期 | 事件越界，或完整 `-1～4 s` 窗口碰到坏段/接缝 | 查看 `stage4_epoch_report.json` 的拒绝原因，不要直接恢复被拒绝 Epoch |
| HDF5 与报告哈希不一致 | 发布过程中中断、文件被外部修改或并发运行 | 停止其他进程，删除/隔离不一致输出后重新运行阶段 4 |
| `ws.py` 无法读取 | 缺少 `clean_eeg`、`labels`，或形状/标签数量不一致 | 使用阶段 4 的 ws 输出，并先运行 `validate_clean_hdf5` |
| FBCSP 阶段 4 提示输出已存在 | `FBCSP_OVERWRITE=False` 的保护机制 | 输入和设置未变则跳过阶段 4；确需重做时审核路径并显式允许覆盖 |
| FBCSP 置换提示缺少 `--observed` | CLI 必填参数未提供 | 指向新分类的 `classification_result.json`，先加 `--validate-only` 核验 |
| 分类/置换提示目录已存在 | 新实验拒绝复用已有输出目录 | 改用尚不存在的新目录；中断续跑使用 `--resume`，不要覆盖检查点 |
| FBCSP 协议或文件哈希不匹配 | 混用了旧实验或发布后文件发生变化 | 重新分类并开始新置换，不修改协议标识绕过检查 |

如果发布失败，优先查看终端异常和对应 JSON 凭据。不要把临时目录中的文件直接
当作正式结果使用；阶段代码设计为验证成功后才替换配置中的正式路径。

---

## 12. 测试和辅助检查

在不需要真实数据的情况下，可以运行当前阶段的回归测试：

```bat
:: 请先在项目根目录打开终端
python -m unittest -v ^
    EEG_preprocessing_test ICA_processing_test EEG_processing_test main_test ^
    ws_test permutation_test grouped_search_test public_paths_test ^
    fbcsp_test fbcsp_grouped_search_test fbcsp_stage4_test fbcsp_permutation_test
```

也可以检查所有 Python 文件是否能编译：

```bat
python -m compileall -q .
```

项目中的辅助工具包括：

| 脚本 | 用途 |
| --- | --- |
| `Viewing_HDF5_Structure.py` | 查看 HDF5 数据集结构 |
| `XML_Metadata_Viewer.py` | 查看 XML 元数据 |
| `Core_EEG_data_acquisition.py` | 读取原始 EEG、事件和 Session 信息 |
| `mapping_dictionary.py` | 查看事件 TypeID 映射 |
| `eeg_viewer.py` | 浏览 EEG 波形 |

### 已处理 HDF5 的独立拼接工具

`processed_data_concatenation.py` 只用于已经分段/清洗好的 HDF5，不处理原始连续
HDF5。输入文件必须包含 `clean_eeg`、`labels`、`channel_names`，且通道数、时间点
数、dtype、通道名、采样率必须一致；它只沿 `clean_eeg` 的第 0 维（Epoch 维）拼接。

```bat
python processed_data_concatenation.py --dry-run
python processed_data_concatenation.py --overwrite
```

该工具与阶段 1/2 的原始连续 Raw 拼接不同，不能用来替代 `EEG_preprocessing.py`。

---

## 13. 当前限制

- 阶段 4 的滤波、事件映射、完整窗口、分类窗口、基线、输出单位和 dtype 均由
  `config.py` 控制；采样点数根据输入采样率与时间窗自动确定，无需同步修改实现。
- 阶段 2 和阶段 3 clean 输出当前只支持单文件 FIF，不支持发布分卷 FIF。
- 阶段 3 的 ICA 成分排除完全依赖人工判断；程序不提供伪迹自动分类结论。
- `standard_1020` 是标准电极位置模板，不等同于每名被试的个体化电极测量位置。
- 阶段 4 是离线零相位滤波，不应直接解释为实时、因果的在线处理。
- `ws.py` 的分类代码是独立消费端，默认只验证 clean HDF5 的基本结构，不替代
  阶段 4 的来源和审核报告校验。

完成原单频带阶段 4 后，真正用于分类的最终文件是：

```text
cache/stage4_epochs_ws.hdf5
```

与它配套的审计依据是：

```text
reports/stage4_epoch_report.json
```

二者应与阶段 3 的 `applied_components.json` 一起保存，才能完整追溯从原始数据
到最终分类输入的处理过程。

FBCSP 分支则使用 `cache/stage4_fbcsp_epochs_ws.hdf5` 及同源 baseline，
配套 `reports/stage4_fbcsp_epoch_report.json`；不要与原单频带阶段 4 的文件混用。
