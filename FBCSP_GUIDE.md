# FBCSP-SVM 实验入口（不含 mRMR）

## 1. 方法及兼容边界

阶段 3 不变。新增阶段 4 入口位于
`fbcsp_stage4.py`：从正式 ICA clean Raw
直接分别滤出五个子频带，并另行生成同源宽带 baseline。原
`EEG_processing.py`、`main.py` 和 `ws.py` 的单频带入口继续可用。
运行旧阶段4入口不会自动生成新产物；运行新入口也不会覆盖旧阶段4/CSP结果。

默认五带：8–12、12–16、16–20、20–24、24–30 Hz。参考宽带由
`STAGE4_FILTER_L_FREQ/H_FREQ` 指定，默认 8–30 Hz。子带必须落在参考范围内。
所有分支直接从同一个 clean Raw 滤波，不串联宽带和子带滤波。
参考分支以原阶段4完整窗口和注释规则筛选试次；各带使用同一事件集合，
并再次检查注释和边界。若带间试次不一致，拒绝发布，不能静默按行拼接。

FBCSP 输入为 `(epochs,bands,channels,times)`，独立于旧三维契约。
每带在当前训练折内拟合一个 MNE CSP，拼接特征后接 StandardScaler 与线性 SVM。
`log=False` 沿用 MNE 的非对数标准化平均功率；`log=True` 为对数平均功率。
没有额外归一化方差定义、mRMR、PCA 或自动频带选择。

候选从 `ws.DEFAULT_PARAM_GRID` 派生并使用相同验证：

| 参数 | 候选 |
| --- | --- |
| reg | 0.1、ledoit_wolf |
| n_components | 2、4（每个频带） |
| log | True |
| SVM C | 0.1、1、10 |

通道数不少于4时总计12个候选。一个候选统一应用于所有频带，但各带空间滤波器独立。
两模型使用10×5嵌套分层CV、shuffle=False、accuracy选参和相同的外层测试索引。
默认采用 grouped 内层搜索，显式保留 MNE `cov_est="concat"`，不增加协方差缓存，
不使用 Pipeline 磁盘变换缓存。每个内层训练折、每个 reg、每带只拟合一次 CSP；
每带先用 `filters_[:n_components]` 投影和生成特征，再按固定带序拼接。
`log=True` 取对数；`log=False`（自定义网格仍支持）按该训练折的功率均值/标准差
归一化。各 `(reg, n_components, log)` 独立生成并缩放特征，不跨 log 候选复用；
只有 SVM C 不同时复用同组已生成的特征。Scaler 仅拟合内层训练特征。
内层索引只用于外层训练子集；选参后整个 Pipeline 在全外层训练集重新拟合。

当前 FBCSP 协议标识为 `nested_fbcsp_v2_grouped_concat`，配对 CSP 为
`nested_csp_paired_v2_grouped_concat`。默认五带10×5搜索共拟合
`10 × (5 × 2 × 5 + 5) = 550` 个带级 CSP（包括外层重拟合）。
`cache_transformers` 旧 API 参数仅为兼容保留，在 FBCSP 入口不再启用缓存。
`search_engine="reference"` 仅供 Python API 使用原无缓存 GridSearchCV 做等价对照。

## 2. 环境和安全默认值

使用 `requirements.txt` 的已验证环境
（Python 3.9.25、MNE 1.8.0、NumPy 2.0.1、scikit-learn 1.6.1）。以下命令中的
`python` 应指向该环境。不要直接使用缺少 MNE 的系统/base Python。

本文路径均相对于项目根目录，命令在项目根目录执行。
本地配置为 `config.py`；公开配置模板为 `config.example.py`。
已有本地配置时不要用模板覆盖，只补齐缺少的 FBCSP 配置项。
新增 `FBCSP_BANDS`、full/ws/baseline/report 四个输出路径、`FBCSP_OVERWRITE`。
`FBCSP_OVERWRITE=False` 默认拒绝覆盖。历史 CSP 输入和结果路径始终受保护。
如确实需要重做新阶段4发布，审核路径后显式设置 True；重新发布后旧分类/置换结果
不再对应新输入，不能直接续跑。

阶段4采用临时文件、回读、源文件复核、逐文件替换、异常回滚、报告最后发布。
不是跨文件系统事务；禁止并发发布/读取，断电或强杀后须核对报告哈希或重新发布。

## 3. 运行顺序

先完成现有阶段3 fit/review/preview/apply，保证清洗文件、manifest、apply凭据一致。
无需先运行旧阶段4，新入口会自己生成配对baseline。
如果阶段3/4输入与发布文件均未改变，不需要重做这些阶段，只需重新分类。
历史72候选分类、置换分布及旧协议检查点不适用于当前实验；不得修改其协议标识
后继续使用。新协议会拒绝这些旧观测记录。先在新目录完成新分类，随后用该分类
JSON在另一新目录开始完整置换；不要续跑旧检查点。

```bat
python fbcsp_stage4.py
python fbcsp_main.py --n-jobs 1
```

新阶段4输出：

- `cache/stage4_fbcsp_epochs_full.hdf5`
- `cache/stage4_fbcsp_epochs_ws.hdf5`
- `cache/stage4_fbcsp_baseline_ws.hdf5`
- `reports/stage4_fbcsp_epoch_report.json`

分类入口默认同时训练 FBCSP 与同源单频带 CSP，保存各折索引、最优参数、预测、
decision function、OOF指标及配对准确率差。`--fbcsp-only` 可跳过配对CSP。
结果写入 `FBCSP_result/` 下新建的时间戳目录；
也可用 `--output-dir` 指定**尚不存在**的独立目录。JSON 是最后发布的完成标记，
MAT 同时包含完整 JSON 记录。失败时目录可能留下不完整辅助文件，不得作为观测结果。

`accuracy` 是外层折准确率等权平均，与原baseline一致；
`metrics.oof_accuracy` 是所有外层预测汇总的准确率。折大小不等时二者可能不同。
报告含 balanced accuracy、macro-F1、各类召回率与混淆矩阵；
不把不同折未经校准的 decision 分数混合计算一个 AUC。

## 4. FBCSP 专用置换检验

入口为 `fbcsp_permutation.py`。
将 `--observed` 指向实际新分类目录中的 `classification_result.json`。
先执行该入口的 `--help` 查看参数；可先加 `--validate-only` 只读核验观测结果。
默认 `--n-permutations 100 --seed 42`，正式次数须预先决定，不能根据中途p值停止。
`--observed` 必填，直接运行脚本而不提供参数不会启动实验。CMD 示例（先替换目录名）：

```bat
set "OBSERVED=FBCSP_result\your_classification_run\classification_result.json"
set "PERM_OUTPUT=FBCSP_result\your_new_permutation_run"
python fbcsp_permutation.py --observed "%OBSERVED%" --validate-only
python fbcsp_permutation.py --observed "%OBSERVED%" --n-permutations 100 --seed 42 --output-dir "%PERM_OUTPUT%"
```

`PERM_OUTPUT` 必须尚不存在，不要预先创建。检查点在该目录的
`permutation_results.json`，每完成一次置换保存一次。
`--resume` 指向实际检查点 `permutation_results.json`，N和seed必须与原任务一致。
续跑不能同时提供 `--output-dir`：

```bat
python fbcsp_permutation.py --observed "%OBSERVED%" --resume "%PERM_OUTPUT%\permutation_results.json" --n-permutations 100 --seed 42
```

续跑带独占锁；异常中断的残留锁仅在确认没有运行进程后人工处理。

每次对Epoch标签作一次全局置换，五带共享同一标签向量。每次按置换后的标签重新
分层划分，完整重跑原观测实验的候选网格、各带CSP、Scaler和SVM。固定无标签滤波
不重跑，真实标签拟合的CSP不复用。p=(1+随机准确率>=观测准确率的次数)/(N+1)。
检查点绑定所有发布文件哈希、观测JSON哈希、版本、网格、划分规则和随机序列。
不会接受旧CSP置换分布，也不会接受CSP的MAT作为FBCSP观测输入。
当前置换协议为 `permutation_fbcsp_v2_grouped_concat`；100 次置换的最小可报告
p 值为 `1/101 ≈ 0.0099`，不应报告为 0。

本检验要求标签在试次间可交换；若存在session/run依赖，应先设计受限置换，
本入口尚不提供这种模式。模型相对随机标签显著不意味着比另一模型显著更好。
上游ICA、人工审核及参考在分类CV之前完成，本检验不声称整条预处理都折内拟合。

## 5. 测试（仅临时合成 EEG）

在项目根目录执行：

```bat
python -m unittest -v fbcsp_grouped_search_test fbcsp_test fbcsp_stage4_test fbcsp_permutation_test
python -m unittest -v EEG_processing_test ws_test grouped_search_test permutation_test
```

测试包含当前完整12候选小型搜索、10×5小网格、频带独立拟合、逐训练索引追踪、低秩、
原 Pipeline/分组搜索逐候选折分数、两种log特征、均分并列首选、外层预测与decision对照、
并行一致性、实际阶段3凭据→阶段4发布、频响、边界隔离、发布回滚、输入损坏、
配对预测、置换完整重拟合、随机序列及断点续跑。不需要也不改写真实EEG。
完整真实数据12候选×10×5加置换仍可能耗时较长，应先估算耗时再提交正式实验。