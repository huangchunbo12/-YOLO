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

# ==========================================
# 1. 实验环境全局配置
# ==========================================
DATA_CONFIG = "solar.yaml"
BASE_WEIGHT = "yolo26n.pt"
TEMPLATE_PATH = "yolov26-custom.yaml"
GLOBAL_SEED = 42
# ⚠️ Batch=32 占用显存极高，A800(80G) 仅建议 2路并行以防 OOM
MAX_PARALLEL = 4  
LOCAL_RESULTS = "runs/detect"  # 结果根目录
NAS_PATH = "/root/autodl-fs/Solar_Project_Backup"

# 2. 物理隔离：固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # 🚀 开启 CuDNN 优化，释放 GPU 计算潜力

# 3. 日志工程
def setup_worker_logger(exp_name):
    os.makedirs("Solar_Logs", exist_ok=True)
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"Solar_Logs/{exp_name}.log")
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# 4. 动态 YAML 生成
def generate_temp_yaml(spd, lsk, bra, pid):
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    if not spd: content = content.replace("SPDConv, [128]", "Conv, [128, 3, 2]")
    if not lsk: content = content.replace("LSKBlock, [256]", "C3k2, [256, False, 0.25]")
    if not bra: content = content.replace("BRA, [512]", "C3k2, [512, True, 0.5]")

    temp_filename = f"temp_proc{pid}.yaml"
    with open(temp_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return temp_filename

# 5. 核心训练任务进程
def train_worker(task):
    exp, pid = task
    logger = setup_worker_logger(exp['name'])

    # 环境初始化
    os.environ["USE_WIOU"] = "1" if exp.get("wiou", False) else "0"
    set_seed(GLOBAL_SEED)
    torch.cuda.empty_cache()

    logger.info(f"▶️ 启动实验: {exp['name']} | Batch=24 | 并发=4")

    cfg_file = None
    model = None
    try:
        if exp["type"] == "ablation":
            cfg_file = generate_temp_yaml(exp["spd"], exp["lsk"], exp["bra"], pid)
            model = YOLO(cfg_file).load(BASE_WEIGHT)
        else:
            model = YOLO(exp["cfg"])

        # 🚀 核心参数修改：Batch=32 & 最终结果绘图开启
        model.train(
            data=DATA_CONFIG,
            epochs=100,            # 总 100 轮实验
            imgsz=640,
            batch=24,              # 🌟 提升 Batch 大小
            device=0,
            project=LOCAL_RESULTS, 
            name=exp["name"],
            seed=GLOBAL_SEED,
            workers=4,             # 🌟 剥离预处理到 CPU 核心
            amp=True,
            exist_ok=True,
            verbose=False,
            # 🌟 结果绘图与权重保存优化
            save=True,             
            save_period=-1,        # 仅保存最后 100 轮结束时的 best.pt 和 last.pt
            plots=True,            # 🌟 关键：生成训练预览图、PR 曲线、混淆矩阵
            overlap_mask=True,     # 提升分割/检测边缘精度
            val=True               # 开启验证以生成最终指标
        )
        logger.info(f"✅ 实验成功完成: {exp['name']}")

    except Exception as e:
        logger.error(f"❌ 关键错误 {exp['name']}: {str(e)}")
    finally:
        # 显存显式释放
        if model: del model
        if cfg_file and os.path.exists(cfg_file): os.remove(cfg_file)
        torch.cuda.empty_cache()
        gc.collect()
    return f"Finished {exp['name']}"

# 6. NAS 自动化备份逻辑
def backup_to_nas():
    print("\n" + "=" * 50)
    print("📋 开启 NAS 自动化归档流程...")
    if not os.path.exists(NAS_PATH): os.makedirs(NAS_PATH, exist_ok=True)
    if not os.path.exists(LOCAL_RESULTS): return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_base_name = f"Solar_Exp_Backup_{timestamp}"

    try:
        archive_path = shutil.make_archive(archive_base_name, 'tar', root_dir=LOCAL_RESULTS)
        target_file = os.path.join(NAS_PATH, os.path.basename(archive_path))
        shutil.move(archive_path, target_file)
        print(f"✅ 数据备份成功！")
    except Exception as e:
        print(f"❌ 备份失败: {e}")
    print("=" * 50 + "\n")

# ==========================================
# 7. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 构造实验矩阵 (16组消融 + 4组 SOTA)
    experiments = []
    for s in [False, True]:
        for l in [False, True]:
            for b in [False, True]:
                for w in [False, True]:
                    experiments.append({
                        "spd": s, "lsk": l, "bra": b, "wiou": w,
                        "name": f"S{int(s)}L{int(l)}B{int(b)}W{int(w)}",
                        "type": "ablation"
                    })
    for v in [8, 10, 11, 12]:
        experiments.append({"cfg": f"yolo{v}n.pt", "name": f"SOTA_YOLO{v}n", "type": "sota"})

    task_queue = [(exp, i) for i, exp in enumerate(experiments)]

    print(f"🏁 开启 A800 20组全自动消融实验 | Batch: 32 | 诊断图表: 开启")

    # 使用非守护进程模式运行，确保 workers=4 能正常启动
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(train_worker, task): task for task in task_queue}
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"✔️ {result}")
            except Exception as e:
                print(f"⚠️ 任务异常: {e}")

    # 🌟 所有实验结束后归档到 NAS
    backup_to_nas()
    print("🌟 所有消融实验已圆满完成！")