import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

from options import Options
from classification_data_folder import ClassificationDataFolder
from train import get_model   # 复用训练里定义的模型构建函数


def main():
    # ===== 1. 读取配置 =====
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    opt.print_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_dir = opt.test['img_dir']          # 如: ./data/dataset/img/test
    color_label_dir = opt.test['mask_color']  # 彩色标签路径: ./data/dataset/mask_color/test
    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']    # 建议改成单个 checkpoint 路径，例如: ./experiments/xxx/checkpoints/checkpoint_best.pth.tar

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ===== 2. 定义数据增强（与训练的 val_transform 保持一致） =====
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ===== 3. 构建测试数据集（按腺体级别） =====
    print("=> Building classification test dataset...")
    test_dataset = ClassificationDataFolder(
        img_dir=img_dir,
        label_dir=color_label_dir,                 # 使用彩色标签目录
        data_transform=test_transform,
        target_size=opt.train['input_size'][0]     # 如 416
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.train['batch_size'],
        shuffle=False,
        num_workers=opt.train['workers'],
        drop_last=False
    )

    # ===== 4. 构建模型并加载权重 =====
    print("=> Creating model...")
    num_classes = opt.model.get('num_classes', 2)
    model = get_model(
        model_name=opt.model['name'],
        num_classes=num_classes,
        pretrained=False  # 测试时不需要加载预训练; 会加载你训练好的权重
    )
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    print(f"=> Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> Loaded checkpoint from epoch {}".format(checkpoint.get('epoch', 'unknown')))

    # ===== 5. 逐腺体测试，统计指标 =====
    all_preds = []
    all_labels = []
    all_probs = []  # 保存预测概率用于计算AUC

    start_time = time.time()
    print("=> Classification test begins. Total glands:", len(test_dataset))

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())  # 保存概率

            if (i + 1) % 10 == 0:
                print(f"\tProcessed {i + 1}/{len(test_loader)} batches")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)  # shape: (N, num_classes)

    # ===== 6. 计算总体精度、macro F1 和分类报告 =====
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print("\n=> Classification results on gland-level:")
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    # 混淆矩阵和详细报告
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # 提取TN, FP, FN, TP（二分类情况）
    if num_classes == 2:
        TN = cm[0, 0]  # 良性被正确预测为良性
        FP = cm[0, 1]  # 良性被错误预测为恶性
        FN = cm[1, 0]  # 恶性被错误预测为良性
        TP = cm[1, 1]  # 恶性被正确预测为恶性
        
        print("\nConfusion Matrix Details:")
        print(f"TN (True Negative, 良性→良性): {TN}")
        print(f"FP (False Positive, 良性→恶性): {FP}")
        print(f"FN (False Negative, 恶性→良性): {FN}")
        print(f"TP (True Positive, 恶性→恶性): {TP}")
        
        # 计算AUC（使用恶性类别的概率）
        auc = roc_auc_score(all_labels, all_probs[:, 1])  # 使用恶性类别的概率
        print(f"\nAUC (Area Under ROC Curve): {auc:.4f}")
    else:
        TN = FP = FN = TP = None
        auc = None
        print("\nNote: TN/FP/FN/TP and AUC are only calculated for binary classification")

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        target_names=[f"class_{i}" for i in range(num_classes)],
        digits=4
    )

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # ===== 7. 将结果保存到文件 =====
    result_file = os.path.join(save_dir, "classification_result.txt")
    with open(result_file, "w") as f:
        f.write(f"Overall accuracy: {acc:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
        
        if num_classes == 2:
            f.write("Confusion Matrix Details:\n")
            f.write(f"TN (True Negative, 良性→良性): {TN}\n")
            f.write(f"FP (False Positive, 良性→恶性): {FP}\n")
            f.write(f"FN (False Negative, 恶性→良性): {FN}\n")
            f.write(f"TP (True Positive, 恶性→恶性): {TP}\n\n")
            f.write(f"AUC (Area Under ROC Curve): {auc:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"=> Test finished. Time: {int(m)}m {s:.1f}s")
    print(f"=> Results saved to: {result_file}")


if __name__ == '__main__':
    main()