# 程序入口 - BCI运动想象分类任务主程序
from pathlib import Path  # 路径处理库
import sys

import scipy.io as sio  # MATLAB文件IO库，用于保存结果为.mat格式

import config  # 项目配置
from ws import train_test  # 导入训练测试函数，执行分类


def run_classification(
    clean_data_path=None,
    k_fold=None,
    need_filter=None,
    result_mat_path=None,
    result_txt_path=None,
):
    """
    执行BCI分类任务：调用训练测试流程，保存分类结果
    
    参数:
        clean_data_path: 已去伪迹的干净数据文件路径，默认使用配置中的路径
        k_fold: 交叉验证折数，默认使用配置中的值
        need_filter: 是否需要带通滤波，默认使用配置中的设置
        result_mat_path: 结果保存的.mat文件路径，用于MATLAB读取，默认使用配置中的路径
        result_txt_path: 结果保存的文本文件路径，方便直接查看，默认使用配置中的路径
    
    返回:
        accuracy: 平均分类准确率
        fold_scores: 每一折的准确率列表
    """
    # 使用默认配置参数
    clean_data_path = Path(clean_data_path or config.CLEAN_OUTPUT_PATH)
    k_fold = k_fold or config.K_FOLD
    need_filter = config.NEED_FILTER if need_filter is None else need_filter
    result_mat_path = Path(result_mat_path or config.CLASSIFICATION_MAT_PATH)
    result_txt_path = Path(result_txt_path or config.CLASSIFICATION_TXT_PATH)

    # 如果需要滤波，设置滤波频段
    filter_band = config.FILTER_BAND if need_filter else None
    # 调用训练测试函数执行分类，得到准确率和各折分数
    accuracy, fold_scores = train_test(
        hdf5_path=str(clean_data_path),
        k_fold=k_fold,
        filter_band=filter_band,
    )

    # 保存本次运行配置信息到结果中，方便后续追溯
    run_config = {
        "clean_data_path": str(clean_data_path),
        "k_fold": k_fold,
        "need_filter": int(bool(need_filter)),
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
        },
    )

    # 同时保存为文本文件，方便直接查看结果
    with open(result_txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Clean data: {clean_data_path}\n")
        file_obj.write(f"Accuracy: {accuracy:.4f}\n")

    # 在控制台打印结果
    print(f"Average accuracy: {accuracy:.2%}")  # 平均准确率
    print(f"Fold scores: {[f'{score[0]:.2%}' for score in fold_scores]}")  # 各折准确率
    print(f"Saved MAT result to: {result_mat_path}")
    print(f"Saved TXT result to: {result_txt_path}")
    return accuracy, fold_scores


def main():
    """主函数：批量跑完52名被试者，3个session/人，打印进度与结果"""
    config.ensure_project_dirs()

    n_subjects = 52
    n_sessions = 3
    base_dir = config.DATASET_DIR
    total = n_subjects * n_sessions

    subject_avg_accs = []  # 每位被试者的平均准确率
    completed = 0

    print("=" * 70)
    print("开始批量分类：52名被试者 × 3 sessions")
    print("=" * 70)

    for sub_idx in range(1, n_subjects + 1):
        sub_id = f"{sub_idx:03d}"
        session_accs = []

        for sess_idx in range(1, n_sessions + 1):
            sess_id = f"{sess_idx:02d}"
            filename = f"clean_dataset1_processeddata_Sub-{sub_id}_sess-{sess_id}_task-MI_eeg.hdf5"
            hdf5_path = base_dir / filename

            # 文件不存在则跳过
            if not hdf5_path.exists():
                print(f"\n[跳过] 文件不存在: {filename}")
                completed += 1
                continue

            # 调用训练测试，参数全部走config默认，不修改任何算法参数
            accuracy, _ = train_test(
                hdf5_path=str(hdf5_path),
                k_fold=config.K_FOLD,
                filter_band=config.FILTER_BAND if config.NEED_FILTER else None,
            )
            session_accs.append(accuracy)
            completed += 1

            # 进度条可视化
            progress = completed / total
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "-" * (bar_len - filled)
            sys.stdout.write(
                f"\r进度: [{bar}] {completed:>3}/{total} ({progress*100:>5.1f}%) | "
                f"Sub-{sub_id} Sess-{sess_id} Acc={accuracy:.2%}"
            )
            sys.stdout.flush()

        # 打印该被试者平均准确率
        if session_accs:
            sub_avg = sum(session_accs) / len(session_accs)
            subject_avg_accs.append(sub_avg)
            sys.stdout.write(
                f"\n  → 被试者 Sub-{sub_id} 平均准确率: {sub_avg:.2%}（基于 {len(session_accs)} 个 session）\n"
            )
        else:
            sys.stdout.write(f"\n  → 被试者 Sub-{sub_id} 无有效 session 数据\n")

        sys.stdout.flush()

    print("=" * 70)
    if subject_avg_accs:
        final_avg = sum(subject_avg_accs) / len(subject_avg_accs)
        print(f"全部 {len(subject_avg_accs)} 名被试者最终平均准确率: {final_avg:.2%}")
    else:
        print("没有成功处理任何被试者数据。")
    print("=" * 70)

    return subject_avg_accs


if __name__ == "__main__":
    main()