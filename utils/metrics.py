"""
utils/metrics.py
Метрики для оценки качества сегментации
"""

import numpy as np
import torch


class SegmentationMetrics:
    """
    Калькулятор метрик для семантической сегментации
    - Overall Accuracy
    - Per-class Accuracy
    - Mean Class Accuracy
    - IoU (Intersection over Union)
    - Mean IoU
    """
    
    def __init__(self, num_classes=4, ignore_index=-100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Сброс всех счетчиков"""
        self.total_correct = 0
        self.total_seen = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, predictions, targets):
        """
        Обновление метрик
        
        Args:
            predictions: (N,) numpy array или tensor - предсказанные классы
            targets: (N,) numpy array или tensor - истинные классы
        """
        # Конвертация в numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Фильтруем ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Overall accuracy
        correct = (predictions == targets).sum()
        self.total_correct += correct
        self.total_seen += len(targets)
        
        # Confusion matrix
        for t, p in zip(targets, predictions):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[int(t), int(p)] += 1
    
    def get_metrics(self):
        """
        Вычисление всех метрик
        
        Returns:
            dict с метриками
        """
        # Overall Accuracy
        overall_acc = 100.0 * self.total_correct / max(self.total_seen, 1)
        
        # Per-class Accuracy
        class_acc = {}
        for c in range(self.num_classes):
            total_c = self.confusion_matrix[c, :].sum()
            correct_c = self.confusion_matrix[c, c]
            if total_c > 0:
                class_acc[c] = 100.0 * correct_c / total_c
            else:
                class_acc[c] = 0.0
        
        # Mean Class Accuracy
        valid_accs = [acc for acc in class_acc.values() if acc > 0]
        mean_class_acc = np.mean(valid_accs) if valid_accs else 0.0
        
        # IoU per class
        iou_per_class = {}
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            
            if tp + fp + fn > 0:
                iou_per_class[c] = 100.0 * tp / (tp + fp + fn)
            else:
                iou_per_class[c] = 0.0
        
        # Mean IoU
        valid_ious = [iou for iou in iou_per_class.values() if iou > 0]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        
        return {
            'overall_acc': overall_acc,
            'mean_class_acc': mean_class_acc,
            'class_acc': class_acc,
            'mean_iou': mean_iou,
            'iou_per_class': iou_per_class,
            'confusion_matrix': self.confusion_matrix
        }
    
    def print_metrics(self, class_names=None):
        """Красивый вывод метрик"""
        metrics = self.get_metrics()
        
        if class_names is None:
            class_names = {i: f"Class {i}" for i in range(self.num_classes)}
        
        print("\n" + "="*70)
        print("📊 МЕТРИКИ СЕГМЕНТАЦИИ")
        print("="*70)
        
        print(f"\n🎯 Overall Accuracy: {metrics['overall_acc']:.2f}%")
        print(f"📈 Mean Class Accuracy: {metrics['mean_class_acc']:.2f}%")
        print(f"🔷 Mean IoU: {metrics['mean_iou']:.2f}%")
        
        print(f"\n{'Класс':<20} {'Accuracy':<12} {'IoU':<12} {'Count'}")
        print("-"*70)
        for c in range(self.num_classes):
            name = class_names.get(c, f"Class {c}")
            acc = metrics['class_acc'][c]
            iou = metrics['iou_per_class'][c]
            count = self.confusion_matrix[c, :].sum()
            print(f"{name:<20} {acc:>6.2f}%      {iou:>6.2f}%      {count:,}")
        
        print("="*70)


if __name__ == '__main__':
    print("🧪 Тестирование метрик\n")
    
    num_classes = 4
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    # Симуляция предсказаний
    np.random.seed(42)
    for _ in range(10):
        preds = np.random.randint(0, num_classes, size=1000)
        targets = np.random.randint(0, num_classes, size=1000)
        metrics.update(preds, targets)
    
    # Вывод
    class_names = {
        0: "Unclassified",
        1: "Ground",
        2: "Vegetation",
        3: "Building"
    }
    metrics.print_metrics(class_names)