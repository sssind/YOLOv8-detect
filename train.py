# train_v3_fixed.py
from ultralytics import YOLO

# ======================> 参数调整 <===================== #

# 是否导出模型为onnx格式
export_model = False
# YOLOv8模型
YOLO_model = "ultralytics/cfg/models/v8/yolov8-ca.yaml"
# dataset.yaml文件地址
dataset_path = "/root/autodl-tmp/datasets/dataset.yaml"
# 输入图片尺寸
img_size = 1280
# 优化器类型
optimizer = "AdamW"
# 学习批次
batch = 8
# 冻结训练设置
use_freeze_training = True  # 是否使用冻结训练
freeze_epochs = 50  # 冻结训练的轮数
freeze_layers = 15  # 冻结前15层
full_epochs = 600  # 全参数训练轮数


# ======================> 训练模型 <===================== #

def main():
    # 加载预训练模型
    model = YOLO(YOLO_model).load('yolov8m.pt')

    # 第一阶段：冻结训练
    if use_freeze_training:
        print("开始冻结训练阶段...")
        
        # 方法1：手动冻结层（推荐，兼容性更好）
        # 冻结模型的前面层
        for i, (name, param) in enumerate(model.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        print(f"已冻结前 {freeze_layers} 层参数")
        
        # 冻结训练
        freeze_results = model.train(
            # --- 模型与数据配置 ---
            data=dataset_path,

            # --- 训练核心参数 ---
            epochs=freeze_epochs,
            batch=batch,
            imgsz=img_size,
            workers=8,

            # --- 优化器与学习率 ---
            optimizer=optimizer,
            lr0=0.0001,  # 初始学习率（冻结阶段使用较小的学习率）
            lrf=0.01,   # 最终学习率系数

            # --- 正则化与损失 ---
            weight_decay=0.0005,
            box=7.5,    # 边界框损失权重
            cls=0.5,    # 分类损失权重
            dfl=1.5,    # DFL损失权重

            # --- 数据增强 ---
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=0.0,  # 关闭 Mosaic 数据增强
            mixup=0.0,   # 关闭 MixUp 数据增强

            # --- 设备与保存 ---
            device=0,   # 使用GPU
            save=True,  # 保存模型
            save_json=True,  # 保存JSON结果

            # --- 验证配置 ---
            val=True,   # 启用验证

            # --- 早停机制 ---
            patience=30,

            # --- 可视化 ---
            plots=True,

            # --- 其他参数 ---
            deterministic=True,
            # 移除了不支持的参数: freeze, split, amp, save_period
        )

        print("冻结训练阶段完成")
        model2 = YOLO(f"{freeze_results.save_dir}/weights/last.pt")
    else:
        model2 = YOLO(YOLO_model)
    
    # 第二阶段：全参数训练
    print("开始全参数训练阶段...")

    # 解冻所有层
    for param in model2.model.parameters():
        param.requires_grad = True
    print("已解冻所有层参数")

    # 全参数训练
    full_results = model2.train(
        # --- 模型与数据配置 ---
        data=dataset_path,

        # --- 训练核心参数 ---
        epochs=full_epochs,
        batch=batch,
        imgsz=img_size,
        workers=8,

        # --- 优化器与学习率 ---
        optimizer=optimizer,
        lr0=0.0001,   # 全参数训练使用正常学习率
        lrf=0.01,

        # --- 其他参数保持不变 ---
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        device=0,
        save=True,
        save_json=True,
        val=True,
        patience=30,
        plots=True,
        deterministic=True,
    )

    print("全参数训练阶段完成")

    # 最终模型验证
    print("开始最终模型验证...")
    final_metrics = model2.val()
    print(f"最终模型 mAP50: {final_metrics.box.map50:.4f}")
    print(f"最终模型 mAP50-95: {final_metrics.box.map:.4f}")

    # 导出模型（如果启用）
    if export_model:
        success = model2.export(format="onnx")
        print(f"模型导出 {'成功' if success else '失败'}")


if __name__ == "__main__":
    main()