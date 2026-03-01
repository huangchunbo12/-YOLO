import gc
import logging
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from ultralytics import YOLO

# ==========================================
# 1. 实验环境全局配置
# ==========================================
DATA_CONFIG = "solar.yaml"
BASE_WEIGHT = "yolov8n.pt"
TEMPLATE_PATH = "yolov8-custom.yaml"
GLOBAL_SEED = 42
MAX_PARALLEL = 3  # 🌟 刚好对应你的 3 组任务，全速运行
LOCAL_RESULTS = "runs/detect8"
NAS_PATH = "/root/autodl-fs/Solar_Project_Backup"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_worker_logger(exp_name):
    os.makedirs("Solar_Logs", exist_ok=True)
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"Solar_Logs/{exp_name}.log")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def generate_temp_yaml(spd, pid):
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        content = f.read()

    # 🌟 逻辑：如果 spd 为 False，则将 SPDConv 替换回标准的 Conv 下采样
    if not spd:
        content = content.replace("SPDConv, [64]", "Conv, [64, 3, 2]")
        content = content.replace("SPDConv, [128]", "Conv, [128, 3, 2]")

    temp_filename = f"temp_v8_proc{pid}.yaml"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(content)
    return temp_filename


def train_worker(task):
    exp, pid = task
    logger = setup_worker_logger(exp["name"])

    # 🌟 核心修改：通过环境变量激活 WIoU 损失函数
    os.environ["USE_WIOU"] = "1" if exp.get("wiou", False) else "0"
    set_seed(GLOBAL_SEED)
    torch.cuda.empty_cache()

    logger.info(f"▶️ 启动实验: {exp['name']} | Batch=24 | GPU: A800")

    cfg_file = None
    model = None
    try:
        cfg_file = generate_temp_yaml(exp["spd"], pid)
        model = YOLO(cfg_file).load(BASE_WEIGHT)

        model.train(
            data=DATA_CONFIG,
            epochs=100,
            imgsz=640,
            batch=24,
            device=0,
            project=LOCAL_RESULTS,
            name=exp["name"],
            seed=GLOBAL_SEED,
            workers=4,
            amp=True,
            exist_ok=True,
            plots=True,
            val=True,
        )
        logger.info(f"✅ 实验成功完成: {exp['name']}")

    except Exception as e:
        logger.error(f"❌ 关键错误 {exp['name']}: {e!s}")
    finally:
        if model:
            del model
        if cfg_file and os.path.exists(cfg_file):
            os.remove(cfg_file)
        torch.cuda.empty_cache()
        gc.collect()
    return f"Finished {exp['name']}"


def backup_to_nas():
    print("\n" + "=" * 50)
    print("📋 开启 NAS 自动化归档流程...")
    if not os.path.exists(NAS_PATH):
        os.makedirs(NAS_PATH, exist_ok=True)
    if not os.path.exists(LOCAL_RESULTS):
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_base_name = f"Solar_v8_Exp_{timestamp}"

    try:
        archive_path = shutil.make_archive(archive_base_name, "tar", root_dir=LOCAL_RESULTS)
        target_file = os.path.join(NAS_PATH, os.path.basename(archive_path))
        shutil.move(archive_path, target_file)
        print("✅ 数据备份至 NAS 成功！")
    except Exception as e:
        print(f"❌ 备份失败: {e}")
    print("=" * 50 + "\n")


# ==========================================
# 7. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 🌟 按照你的要求，仅保留这三组实验
    experiments = [
        {"spd": True, "wiou": False, "name": "YOLOv8_S1W0"},  # 实验 1: s1w0
        {"spd": False, "wiou": True, "name": "YOLOv8_S0W1"},  # 实验 2: s0w1
        {"spd": True, "wiou": True, "name": "YOLOv8_S1W1"},  # 实验 3: s1w1
    ]

    task_queue = [(exp, i) for i, exp in enumerate(experiments)]

    print(f"🏁 开启 YOLOv8 精简消融实验 | 并发: {MAX_PARALLEL}")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(train_worker, task): task for task in task_queue}
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"✔️ {result}")
            except Exception as e:
                print(f"⚠️ 任务异常: {e}")

    backup_to_nas()
    print("🌟 选定的三组消融实验已圆满完成！")
