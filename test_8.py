import os
import csv
import torch
import concurrent.futures
from ultralytics import YOLO

# ==========================================
# 1. 评估环境配置
# ==========================================
# 🌟 注意：确保这里的路径与你 YOLOv8 训练代码中的 project/name 对应
PROJECT_DIR = "runs/detect/runs/detect8" 
DATA_CONFIG = "solar.yaml"
RESULT_FILE = "Final_V8_Ablation_Test_Report.csv"
MAX_PARALLEL = 3  # 🌟 刚好对应你的 3 组消融任务
CLASS_NAMES = ['Dust', 'Non Defective', 'Snow']

# 初始化表格字段
base_fields = [
    'Experiment', 'SPD_Switch', 'WIoU_Switch', 'mAP50', 'mAP50-95',
    'Precision', 'Recall', 'Inference_Time(ms)'
]
for cls in CLASS_NAMES:
    cls_name = cls.replace(' ', '_')
    base_fields.extend([f'{cls_name}_mAP50', f'{cls_name}_Precision'])
FIELDS = base_fields

def test_worker(task):
    exp, pid = task
    # 路径构造：runs/detect/YOLOv8_S1W0/weights/best.pt
    weight_path = os.path.join(PROJECT_DIR, exp['name'], "weights", "best.pt")
    
    if not os.path.exists(weight_path):
        print(f"⚠️ 跳过: {exp['name']} (权重未找到)")
        return None

    # 环境模拟：确保 WIoU 开关与实验组对齐
    os.environ["USE_WIOU"] = "1" if exp["wiou"] else "0"

    try:
        model = YOLO(weight_path)
        
        # 🌟 执行测试集评估 (split='test')
        # 设置 batch=1 以获取最准确的单帧推理延迟（Latency）数据
        results = model.val(
            data=DATA_CONFIG,
            split='test',
            imgsz=640,
            batch=1, 
            device=0,
            plots=True,      # 生成测试集专属混淆矩阵和 PR 曲线
            project=PROJECT_DIR,
            name=f"{exp['name']}_Test_Eval",
            verbose=False
        )

        # 提取全局指标 (适配 YOLOv8 属性访问)
        metrics = {
            'Experiment': exp['name'],
            'SPD_Switch': "ON" if exp['spd'] else "OFF",
            'WIoU_Switch': "ON" if exp['wiou'] else "OFF",
            'mAP50': round(results.box.map50, 4),
            'mAP50-95': round(results.box.map, 4),
            'Precision': round(results.box.mp, 4),
            'Recall': round(results.box.mr, 4),
            'Inference_Time(ms)': round(results.speed['inference'], 2)
        }

        # 提取类别级指标
        for i, cls_name in enumerate(CLASS_NAMES):
            cls_name_safe = cls_name.replace(' ', '_')
            metrics[f'{cls_name_safe}_mAP50'] = round(results.box.ap50[i], 4)
            metrics[f'{cls_name_safe}_Precision'] = round(results.box.p[i], 4)

        print(f"✅ 测试完成: {exp['name']} | mAP50: {metrics['mAP50']}")
        
        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"❌ 评估出错 {exp['name']}: {str(e)}")
        return None

if __name__ == "__main__":
    # 🌟 构造你指定的三组核心消融组
    experiments = [
        {"spd": True,  "wiou": False, "name": "YOLOv8_S1W0"}, # s1w0
        {"spd": False, "wiou": True,  "name": "YOLOv8_S0W1"}, # s0w1
        {"spd": True,  "wiou": True,  "name": "YOLOv8_S1W1"}  # s1w1
    ]

    print(f"🚀 开始 A800 批量评估（3组核心消融），结果将汇总至 {RESULT_FILE}")

    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        task_list = [(exp, i) for i, exp in enumerate(experiments)]
        future_to_task = {executor.submit(test_worker, task): task for task in task_list}
        
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            if res: all_results.append(res)

    # 保存最终汇总报告
    with open(RESULT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n🌟 评估已圆满完成！汇总报告: {os.path.abspath(RESULT_FILE)}")