# advanced_metrics_analysis.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
import numpy as np


class YOLOEvaluator:
    def __init__(self, model_path, data_yaml):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = YOLO(model_path)
        self.metrics = None

    def comprehensive_evaluation(self, split='test'):
        """执行全面的模型评估"""

        print("🔍 执行全面模型评估...")

        # 基础验证
        self.metrics = self.model.val(
            data=self.data_yaml,
            split=split,
            save_json=True,
            plots=True,
            conf=0.001,
            iou=0.6,
            device=0
        )

        # 生成详细报告
        self._generate_detailed_report()

        return self.metrics

    def _generate_detailed_report(self):
        """生成详细评估报告"""

        print("\n" + "=" * 80)
        print("📈 详细评估报告")
        print("=" * 80)

        # 1. 基础性能指标
        self._print_basic_metrics()

        # 2. 各类别性能
        self._print_class_metrics()

        # 3. 混淆矩阵分析
        self._analyze_confusion_matrix()

        # 4. 性能建议
        self._provide_recommendations()

    def _print_basic_metrics(self):
        """打印基础性能指标"""

        print("\n📊 基础性能指标:")
        print(f"  • mAP@0.5:       {self.metrics.box.map50:.4f}")
        print(f"  • mAP@0.5:0.95:  {self.metrics.box.map:.4f}")
        print(f"  • 精确率:        {self.metrics.box.precision:.4f}")
        print(f"  • 召回率:        {self.metrics.box.recall:.4f}")

        # 计算F1分数
        precision = self.metrics.box.precision
        recall = self.metrics.box.recall
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  • F1分数:        {f1:.4f}")

        # 推理速度
        if hasattr(self.metrics, 'speed'):
            speed = self.metrics.speed
            print(f"  • 推理速度:      {speed.get('inference', 0):.2f} ms/图像")
            print(f"  • FPS:          {1000 / speed.get('inference', 1):.2f}")

    def _print_class_metrics(self):
        """打印各类别指标"""

        if hasattr(self.metrics.box, 'aps') and self.metrics.box.aps is not None:
            print(f"\n🎯 各类别AP (共{len(self.metrics.box.aps)}个类别):")

            # 按AP值排序
            sorted_aps = sorted(enumerate(self.metrics.box.aps),
                                key=lambda x: x[1], reverse=True)

            for i, (class_id, ap) in enumerate(sorted_aps):
                status = "✅" if ap > 0.5 else "⚠️" if ap > 0.3 else "❌"
                print(f"  {status} 类别 {class_id}: AP = {ap:.4f}")

                # 只显示前10个和后10个
                if i == 9 and len(sorted_aps) > 20:
                    print(f"  ... 省略中间 {len(sorted_aps) - 20} 个类别 ...")
                    break

    def _analyze_confusion_matrix(self):
        """分析混淆矩阵（如果可用）"""

        # 检查是否有混淆矩阵数据
        results_dir = self.metrics.save_dir
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')

        if os.path.exists(cm_path):
            print(f"\n🔀 混淆矩阵已生成: {cm_path}")
        else:
            print("\n🔀 混淆矩阵: 未生成或不可用")

    def _provide_recommendations(self):
        """基于指标提供改进建议"""

        print("\n💡 性能分析与建议:")

        map50 = self.metrics.box.map50
        precision = self.metrics.box.precision
        recall = self.metrics.box.recall

        if map50 < 0.3:
            print("  ❌ 模型性能较差，建议:")
            print("     - 检查数据质量")
            print("     - 增加训练轮次")
            print("     - 调整数据增强")
        elif map50 < 0.6:
            print("  ⚠️ 模型性能中等，建议:")
            print("     - 微调超参数")
            print("     - 增加困难样本")
            print("     - 尝试不同尺寸训练")
        else:
            print("  ✅ 模型性能良好!")

        # 精确率-召回率平衡建议
        if precision > recall + 0.2:
            print("  📉 召回率偏低，建议降低置信度阈值")
        elif recall > precision + 0.2:
            print("  📈 精确率偏低，建议提高置信度阈值")

    def compare_with_baseline(self, baseline_metrics_path):
        """与基线模型比较"""

        try:
            with open(baseline_metrics_path, 'r') as f:
                baseline_metrics = json.load(f)

            current_map = self.metrics.box.map
            baseline_map = baseline_metrics.get('map', 0)

            improvement = current_map - baseline_map

            print(f"\n📊 与基线比较:")
            print(f"  • 当前mAP: {current_map:.4f}")
            print(f"  • 基线mAP: {baseline_map:.4f}")
            print(f"  • 改进:    {improvement:+.4f} ({improvement / baseline_map * 100:+.1f}%)")

        except Exception as e:
            print(f"⚠️ 基线比较失败: {e}")


def main():
    # 配置
    evaluator = YOLOEvaluator(
        model_path='runs/detect/train3/weights/best.pt',
        data_yaml='datasets/dataset.yaml'
    )

    # 执行全面评估
    metrics = evaluator.comprehensive_evaluation(split='test')

    # 保存详细报告
    report_path = 'runs/val/detailed_evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        import sys
        from io import StringIO

        # 重定向输出到文件
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        evaluator._generate_detailed_report()

        report_content = sys.stdout.getvalue()
        sys.stdout = old_stdout

        f.write(report_content)

    print(f"\n💾 详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()