import concurrent.futures
import csv
import os

import torch

from ultralytics import YOLO

# 1. 路径与配置
PROJECT_DIR = "runs/detect/runs/detect"
DATA_CONFIG = "solar.yaml"
RESULT_FILE = "Final_Ablation_Test_Report.csv"
MAX_PARALLEL = 4  # A800 4路并行最优
CLASS_NAMES = ["Dust", "Non Defective", "Snow"]  # 从你的yaml中获取

# 2. 初始化 CSV 表头（扩展为包含每个类别的指标）
base_fields = ["Experiment", "WIoU_Switch", "mAP50", "mAP50-95", "Precision", "Recall", "Fitness", "Inference_Time(ms)"]
# 为每个类别添加指标列
for cls in CLASS_NAMES:
    cls_name = cls.replace(" ", "_")
    base_fields.extend([f"{cls_name}_mAP50", f"{cls_name}_mAP50-95", f"{cls_name}_Precision", f"{cls_name}_Recall"])
FIELDS = base_fields


def test_worker(task):
    exp, pid = task
    # 路径构造
    weight_path = os.path.join(PROJECT_DIR, exp["name"], "weights", "best.pt")

    # 增加last.pt兜底
    if not os.path.exists(weight_path):
        weight_path = os.path.join(PROJECT_DIR, exp["name"], "weights", "last.pt")
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过: {exp['name']} (未找到best/last权重文件)")
            return None

    # WIoU开关
    wiou_switch = "N/A"
    if "wiou" in exp:
        os.environ["USE_WIOU"] = "1" if exp["wiou"] else "0"
        wiou_switch = os.environ["USE_WIOU"]

    try:
        # 加载模型
        model = YOLO(weight_path)

        # 测试集评估（开启可视化，输出到独立文件夹）
        test_output_dir = os.path.join("runs/detect", f"{exp['name']}_test")
        results = model.val(
            data=DATA_CONFIG,
            split="test",
            imgsz=640,
            batch=24,
            device=0,
            plots=True,  # 开启可视化输出
            project="runs/detect",
            name=f"{exp['name']}_test",  # 每个实验独立输出文件夹
            verbose=True,  # 开启详细输出，方便调试
            workers=0,
        )

        # 提取全局指标
        global_metrics = {
            "Experiment": exp["name"],
            "WIoU_Switch": wiou_switch,
            "mAP50": round(results.box.map50, 4),
            "mAP50-95": round(results.box.map, 4),
            "Precision": round(results.box.mp, 4),
            "Recall": round(results.box.mr, 4),
            "Fitness": round(results.fitness, 4),
            "Inference_Time(ms)": round(results.speed["inference"], 2),
        }

        # 提取每个类别的指标
        per_class_metrics = []
        for i, cls_name in enumerate(CLASS_NAMES):
            cls_name_safe = cls_name.replace(" ", "_")
            cls_metrics = {
                "Class": cls_name,
                "mAP50": round(results.box.ap50[i].item(), 4),
                "mAP50-95": round(results.box.ap[i].item(), 4),
                "Precision": round(results.box.p[i].item(), 4),
                "Recall": round(results.box.r[i].item(), 4),
            }
            per_class_metrics.append(cls_metrics)
            # 填充到全局汇总字典
            global_metrics[f"{cls_name_safe}_mAP50"] = cls_metrics["mAP50"]
            global_metrics[f"{cls_name_safe}_mAP50-95"] = cls_metrics["mAP50-95"]
            global_metrics[f"{cls_name_safe}_Precision"] = cls_metrics["Precision"]
            global_metrics[f"{cls_name_safe}_Recall"] = cls_metrics["Recall"]

        # 保存每个实验的独立输出文件
        os.makedirs(test_output_dir, exist_ok=True)
        # 1. 保存类别级指标CSV
        per_class_csv = os.path.join(test_output_dir, "per_class_metrics.csv")
        with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Class", "mAP50", "mAP50-95", "Precision", "Recall"])
            writer.writeheader()
            writer.writerows(per_class_metrics)
        # 2. 保存汇总指标CSV
        summary_csv = os.path.join(test_output_dir, "summary_metrics.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(global_metrics)

        print(f"✅ 完成测试: {exp['name']} | mAP50: {global_metrics['mAP50']}")

        # 释放显存
        del model
        torch.cuda.empty_cache()
        return global_metrics

    except Exception as e:
        print(f"❌ 测试出错 {exp['name']}: {e!s}")
        if "model" in locals():
            del model
            torch.cuda.empty_cache()
        return None


if __name__ == "__main__":
    # 构造消融实验
    ablation_experiments = []
    for use_spd in [False, True]:
        for use_lsk in [False, True]:
            for use_bra in [False, True]:
                for use_wiou in [False, True]:
                    ablation_experiments.append(
                        {"wiou": use_wiou, "name": f"S{int(use_spd)}L{int(use_lsk)}B{int(use_bra)}W{int(use_wiou)}"}
                    )

    # 构造SOTA模型
    sota_experiments = []
    for version in [8, 10, 11, 12]:
        sota_experiments.append({"name": f"SOTA_YOLO{version}n"})

    # 合并所有实验
    experiments = ablation_experiments + sota_experiments
    task_queue = [(exp, i) for i, exp in enumerate(experiments)]

    print(f"🚀 开始批量测试 20 组权重（16组消融+4组SOTA），结果将保存至 {RESULT_FILE}...")

    # 并行执行
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        future_to_task = {executor.submit(test_worker, task): task for task in task_queue}
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"⚠️ 任务执行异常: {e}")
                all_results.append(None)

    # 保存全局汇总CSV
    final_data = [r for r in all_results if r is not None]
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(final_data)

    print(f"\n🌟 所有测试已完成！报告文件路径: {os.path.abspath(RESULT_FILE)}")
