import os
import torch
import concurrent.futures
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# 核心配置
# ==========================================
PROJECT_DIR = "/root/autodl-tmp/yolo/ultralytics/ultralytics-main/runs/detect/runs/detect8"  
# ✅ 目标YOLOv8n权重路径（重点监控）
YOLO8N_WEIGHT_PATH = "/root/autodl-tmp/yolo/ultralytics/ultralytics-main/runs/detect/runs/detect/SOTA_YOLO8n/weights/best.pt"
MAX_PARALLEL = 3                         
TEST_IMAGES_DIR = Path("/root/autodl-tmp/yolo/Solar_Filtered_3_Classes/test1/images")

def visualize_worker(task):
    """仅负责生成单组实验的全量可视化结果（带详细调试日志）"""
    exp, pid = task
    exp_name = exp['name']
    
    # ========== 调试日志1：打印当前处理的实验名 ==========
    print(f"\n🔍 开始处理：{exp_name} (进程ID: {pid})")
    
    # ========== 核心分支：明确区分权重路径 ==========
    if exp_name == "YOLOv8n":
        # 加载YOLOv8n训练后的best.pt
        weight_path = YOLO8N_WEIGHT_PATH
        # ========== 调试日志2：确认进入YOLOv8n分支 ==========
        print(f"📌 检测到YOLOv8n实验，使用权重路径：{weight_path}")
        weight_search_key = "SOTA_YOLO8n/weights/best.pt"  # 精准查找关键词
    else:
        # 原有消融实验权重
        weight_path = os.path.join(PROJECT_DIR, exp_name, "weights", "best.pt")
        weight_search_key = f"{exp_name}/weights/best.pt"
    
    # 校验权重文件是否存在
    if not os.path.exists(weight_path):
        print(f"⚠️ 跳过: {exp_name} (权重文件未找到: {weight_path})")
        print(f"   快速查找命令：find /root/autodl-tmp/yolo/ultralytics/ultralytics-main/ -path '*{weight_search_key}'")
        return None

    try:
        # ========== 调试日志3：确认开始加载模型 ==========
        print(f"📥 开始加载模型：{weight_path}")
        # 加载模型
        model = YOLO(weight_path)
        # ========== 调试日志4：确认模型加载成功 ==========
        print(f"✅ 模型加载成功：{exp_name} (模型类型: {type(model)})")
        
        # 生成可视化（确保置信度显示）
        results = model.predict(
            source=str(TEST_IMAGES_DIR),  
            save=True,                   # 保存检测结果
            save_txt=False,              # 不保存txt标签
            save_conf=True,              # 保存时显示置信度
            show_conf=True,              # 显式开启置信度显示
            imgsz=640,                   # 推理尺寸
            batch=1,                     # 批量大小
            device=0,                    # GPU编号
            verbose=False,               # 关闭冗余日志
            project=PROJECT_DIR,         # 结果保存根目录
            name=f"{exp_name}_Visual",   # 可视化目录名
            exist_ok=True                # 覆盖已有结果
        )

        # ========== 调试日志5：确认可视化生成完成 ==========
        visual_path = os.path.join(PROJECT_DIR, f"{exp_name}_Visual")
        print(f"📤 可视化结果保存完成：{visual_path} (生成{len(results)}张图片结果)")
        
        # 清理显存（先删results，避免内存泄漏）
        del results
        del model
        torch.cuda.empty_cache()
        
        return visual_path

    except Exception as e:
        print(f"❌ 处理失败 {exp_name}: {str(e)}")
        # 打印完整异常栈（方便定位问题）
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 实验列表
    experiments = [
        {"spd": True,  "wiou": False, "name": "YOLOv8_S1W0"},
        {"spd": False, "wiou": True,  "name": "YOLOv8_S0W1"},
        {"spd": True,  "wiou": True,  "name": "YOLOv8_S1W1"},
        {"name": "YOLOv8n"}  # 重点监控的实验
    ]

    print(f"🚀 开始生成测试集全量可视化结果（含YOLOv8n训练后模型）")
    print(f"==================================================")
    
    # 前置校验（确保基础路径有效）
    if not TEST_IMAGES_DIR.exists():
        print(f"\n❌ 致命错误：测试集目录不存在 → {TEST_IMAGES_DIR}")
        exit(1)
    image_count = len([f for f in TEST_IMAGES_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg','png','jpeg','bmp']])
    if image_count == 0:
        print(f"\n❌ 警告：测试集无图片 → {TEST_IMAGES_DIR}")
        exit(1)
    if not os.path.exists(PROJECT_DIR):
        print(f"\n❌ 致命错误：消融实验权重目录不存在 → {PROJECT_DIR}")
        exit(1)
    if not os.path.exists(YOLO8N_WEIGHT_PATH):
        print(f"\n⚠️ 警告：YOLOv8n权重未找到 → {YOLO8N_WEIGHT_PATH}")
        print(f"   手动验证命令：ls -l {YOLO8N_WEIGHT_PATH}")

    # ========== 关键修改：改用ThreadPoolExecutor（避免进程级参数传递异常） ==========
    print(f"\n📢 并行模式：线程池（确保YOLOv8n权重路径调用成功）")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        task_list = [(exp, i) for i, exp in enumerate(experiments)]
        future_to_task = {executor.submit(visualize_worker, task): task for task in task_list}
        
        # 收集成功结果
        completed_paths = []
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            if res:
                completed_paths.append(res)

    # 最终汇总
    print(f"\n==================================================")
    print(f"🌟 可视化生成完成！")
    if completed_paths:
        print(f"✅ 成功生成的结果路径:")
        for path in completed_paths:
            print(f"   - {path}")
    else:
        print(f"❌ 无有效可视化结果生成！请检查：")
        print(f"   1. 消融实验权重：{PROJECT_DIR}/[实验组名]/weights/best.pt")
        print(f"   2. YOLOv8n权重：{YOLO8N_WEIGHT_PATH}")
        print(f"   3. 手动验证YOLOv8n权重：ls -l {YOLO8N_WEIGHT_PATH}")