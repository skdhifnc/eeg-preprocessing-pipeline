# 程序入口 - BCI运动想象分类任务主程序
from pathlib import Path  # 路径处理库
import json

import h5py
import numpy as np
import scipy.io as sio  # MATLAB文件IO库，用于保存结果为.mat格式

import config  # 项目配置
from ws import (DEFAULT_INNER_K_FOLD, file_sha256, runtime_versions,
                train_test, validate_clean_hdf5)


DEFAULT_K_FOLD = 10
DEFAULT_RESULT_MAT_PATH = (
    config.PROJECT_ROOT / "CSP_result" / "classification_result" / "classification_result.mat"
)
DEFAULT_RESULT_TXT_PATH = config.PROJECT_ROOT / "CSP_result" / "results" / "results.txt"


def run_classification(
    clean_data_path=None,
    k_fold=None,
    need_filter=None,
    result_mat_path=None,
    result_txt_path=None,
    filter_band=None,
    *,
    inner_k_fold=DEFAULT_INNER_K_FOLD,
    param_grid=None,
    n_jobs=1,
):
    """
    执行BCI分类任务：调用训练测试流程，保存分类结果
    
    参数:
        clean_data_path: 默认 config.STAGE4_WS_OUTPUT_PATH，正式阶段 4 分类窗口
        k_fold: 外层折数，默认 DEFAULT_K_FOLD=10，不自动降低折数
        need_filter: 默认不重复滤波；None 时由是否提供 filter_band 决定
        result_mat_path: 默认 DEFAULT_RESULT_MAT_PATH
        result_txt_path: 默认 DEFAULT_RESULT_TXT_PATH
        filter_band: 显式指定的额外带通频段；need_filter=True 时必须提供
        inner_k_fold: 外层训练集内部的调参折数，默认5
        param_grid: None 使用 ws.DEFAULT_PARAM_GRID，否则传入 Pipeline 参数网格
        n_jobs: 内层 GridSearchCV 并行数，默认1
    
    返回:
        accuracy: 平均分类准确率
        fold_scores: 一维 list[float]，每一折的准确率
    """
    clean_data_path = Path(
        config.STAGE4_WS_OUTPUT_PATH if clean_data_path is None else clean_data_path
    ).expanduser().resolve()
    k_fold = DEFAULT_K_FOLD if k_fold is None else k_fold
    result_mat_path = Path(
        DEFAULT_RESULT_MAT_PATH if result_mat_path is None else result_mat_path
    ).expanduser().resolve()
    result_txt_path = Path(
        DEFAULT_RESULT_TXT_PATH if result_txt_path is None else result_txt_path
    ).expanduser().resolve()
    if need_filter is None:
        need_filter = filter_band is not None
    if type(need_filter) is not bool:
        raise TypeError("need_filter must be True, False or None")
    if need_filter and filter_band is None:
        raise ValueError("Stage 4 is already filtered; specify filter_band for additional filtering")
    if not need_filter and filter_band is not None:
        raise ValueError("filter_band conflicts with need_filter=False")
    if clean_data_path == Path(config.STAGE4_FULL_OUTPUT_PATH).resolve():
        raise ValueError("Use stage 4 ws output, not the full audit output")
    # 在训练/写结果前防止误覆盖输入、审核产物或另一个结果文件。
    protected = {clean_data_path, Path(config.STAGE4_WS_OUTPUT_PATH).resolve(),
                 Path(config.STAGE4_FULL_OUTPUT_PATH).resolve(), Path(config.STAGE4_REPORT_PATH).resolve()}
    if (result_mat_path == result_txt_path
            or result_mat_path in protected or result_txt_path in protected):
        raise ValueError("Result paths must be distinct and must not overwrite stage 4 inputs/outputs")
    input_sha256 = file_sha256(clean_data_path)
    validate_clean_hdf5(clean_data_path)
    with h5py.File(clean_data_path, "r") as h5_file:
        if h5_file.attrs.get("stage") != 4:
            raise ValueError("main requires a formal stage 4 HDF5 input")
        values, counts = np.unique(h5_file["labels"][:], return_counts=True)
        summary = {
            "stage": 4,
            "run_id": str(h5_file.attrs["run_id"]),
            "sampling_rate": float(h5_file.attrs["sampling_rate"]),
            "unit": str(h5_file.attrs["unit"]),
            "data_shape": h5_file["clean_eeg"].shape,
            "class_counts_json": json.dumps(dict(zip(map(str, values), map(int, counts)))),
            "label_map_json": str(h5_file.attrs["label_map"]),
            "tmin": float(h5_file.attrs["tmin"]),
            "tmax": float(h5_file.attrs["tmax"]),
        }
    # 外层测试分数用于报告，内层最佳分数仅用于选参和追溯。
    accuracy, fold_scores, cv_details = train_test(
        hdf5_path=str(clean_data_path),
        k_fold=k_fold,
        filter_band=filter_band,
        inner_k_fold=inner_k_fold,
        param_grid=param_grid,
        n_jobs=n_jobs,
        return_details=True,
    )
    if file_sha256(clean_data_path) != input_sha256:
        raise RuntimeError("Classification input changed during fitting; results not published")

    # 保存本次运行配置信息到结果中，方便后续追溯
    run_config = {
        "clean_data_path": str(clean_data_path),
        "input_sha256": input_sha256,
        "cv_protocol": cv_details["cv_protocol"],
        "runtime_versions_json": json.dumps(runtime_versions()),
        "k_fold": k_fold,
        "inner_k_fold": int(inner_k_fold),
        "cv_method": cv_details["cv_method"],
        "shuffle": int(cv_details["shuffle"]),
        "scoring": cv_details["scoring"],
        "n_jobs": int(n_jobs),
        "param_grid_json": json.dumps(cv_details["param_grid"]),
        "need_filter": int(bool(need_filter)),
        "filter_band": [] if filter_band is None else list(map(float, filter_band)),
        **summary,
    }

    # 确保结果目录存在
    result_mat_path.parent.mkdir(parents=True, exist_ok=True)
    result_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # 将结果保存为MATLAB格式文件，包含准确率、各折分数和运行配置
    sio.savemat(
        result_mat_path,
        {
            "accuracy": accuracy,
            "fold_scores": fold_scores,
            "config": run_config,
            # JSON 保留 reg 的数值/字符串混合类型，以及各折长度不同的索引。
            "nested_cv_details_json": json.dumps(cv_details),
            "best_inner_scores": [fold["best_inner_score"] for fold in cv_details["folds"]],
        },
    )

    # 同时保存为文本文件，方便直接查看结果
    with open(result_txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Clean data: {clean_data_path}\n")
        file_obj.write(f"Run ID: {summary['run_id']}\n")
        file_obj.write(f"Data shape: {summary['data_shape']}\n")
        file_obj.write(f"Sampling rate: {summary['sampling_rate']} Hz\n")
        file_obj.write(f"Unit: {summary['unit']}\n")
        file_obj.write(f"Class counts: {summary['class_counts_json']}\n")
        file_obj.write(f"K folds: {k_fold}\n")
        file_obj.write(f"CV method: {cv_details['cv_method']}\n")
        file_obj.write(f"Inner folds: {inner_k_fold}\n")
        file_obj.write(f"Shuffle: {cv_details['shuffle']}\n")
        file_obj.write(f"Parameter grid: {run_config['param_grid_json']}\n")
        file_obj.write(f"Additional filter: {run_config['filter_band']}\n")
        file_obj.write(f"Accuracy: {accuracy:.4f}\n")
        file_obj.write(f"Fold scores: {fold_scores}\n")
        for fold in cv_details["folds"]:
            file_obj.write(f"Outer fold {fold['fold']}: accuracy={fold['outer_score']:.4f}; "
                           f"best inner accuracy={fold['best_inner_score']:.4f}; "
                           f"best params={json.dumps(fold['best_params'])}\n")

    # 在控制台打印结果
    print(f"Average accuracy: {accuracy:.2%}")  # 平均准确率
    print(f"Fold scores: {[f'{score:.2%}' for score in fold_scores]}")  # 各折准确率
    print(f"Saved MAT result to: {result_mat_path}")
    print(f"Saved TXT result to: {result_txt_path}")
    return accuracy, fold_scores


def main():
    """读取正式阶段 4 ws 输出执行分类；只创建分类结果所需目录。"""
    return run_classification()  # 执行分类


if __name__ == "__main__":
    main()
