import os
import torch
import numpy as np
import random
import logging
import gc
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultralytics import YOLO
import ultralytics

# ==========================================
# 1. 测试环境配置
# ==========================================
DATA_CONFIG = "solar.yaml"
# 仅挑选报错的两个 SOTA 模型进行验证
TEST_MODELS = [
    {"cfg": "yolo10n.pt", "name": "Test_v10n"},
    {"cfg": "yolo11n.pt", "name": "Test_v11n"}
]
GLOBAL_SEED = 42
MAX_PARALLEL = 2       # 测试时建议先开2路，观察显存水位
LOCAL_RESULTS = "Test_Results"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True 

def setup_worker_logger(exp_name):
    os.makedirs("Test_Logs", exist_ok=True)
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"Test_Logs/{exp_name}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(fh)
    return logger

def train_worker(task):
    exp, pid = task
    logger = setup_worker_logger(exp['name'])
    
    set_seed(GLOBAL_SEED)
    torch.cuda.empty_cache() # 🌟 启动前清理

    logger.info(f"🚀 测试启动: {exp['name']} | Ultralytics v{ultralytics.__version__}")

    model = None
    try:
        model = YOLO(exp["cfg"])
        # 仅运行 1 轮以验证架构兼容性
        model.train(
            data=DATA_CONFIG,
            epochs=1,              # 🌟 极速测试模式
            imgsz=640,
            batch=24,              # 🌟 沿用你实测稳定的 Batch
            device=0,
            project=LOCAL_RESULTS, 
            name=exp["name"],
            workers=4,             # 🌟 释放 CPU 给预处理
            plots=True,            # 验证 PR 曲线生成逻辑
            save=True,
            verbose=False
        )
        logger.info(f"✅ {exp['name']} 架构解析成功，未报 dim 错误")

    except Exception as e:
        logger.error(f"❌ {exp['name']} 报错: {str(e)}")
    finally:
        # 🌟 核心显存回收逻辑验证
        if model: del model
        torch.cuda.empty_cache()
        gc.collect()
    return f"Test Finished: {exp['name']}"

if __name__ == "__main__":
    print(f"🛠️ 开始环境检测 | 当前版本: {ultralytics.__version__}")
    
    # 构造任务
    task_queue = [(exp, i) for i, exp in enumerate(TEST_MODELS)]

    # 使用非守护进程池
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(train_worker, task): task for task in task_queue}
        for future in as_completed(futures):
            print(f"✔️ {future.result()}")

    print("\n🌟 测试完成！请检查 Test_Logs/ 目录下的日志。")