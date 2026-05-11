import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from datetime import datetime
import re

from options import Options
from classification_data_folder import ClassificationDataFolder
from train import get_model

def test_single_model(model_path, test_loader, opt, device, model_idx, total_models):
    """测试单个分类模型"""
    num_classes = opt.model.get('num_classes', 2)
    
    # 构建模型
    model = get_model(
        model_name=opt.model['name'],
        num_classes=num_classes,
        pretrained=False
    )
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    
    # 加载模型权重
    print(f"=> Loading checkpoint from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"=> Loaded checkpoint from epoch {epoch}")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None
    
    # 开始测试
    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()
    print(f"=> Classification test begins. Total glands: {len(test_loader.dataset)}")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"\tProcessed {i + 1}/{len(test_loader)} batches")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    # 计算指标
    acc = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # 提取TN, FP, FN, TP和AUC（二分类）
    if num_classes == 2:
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
        auc = roc_auc_score(all_labels, all_probs[:, 1])

        # 计算精确率和召回率（针对正类）
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    else:
        TN = FP = FN = TP = None
        auc = None
        precision = recall = None

    # 计算 macro F1（适用于二分类和多分类）
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        target_names=[f"class_{i}" for i in range(num_classes)],
        digits=4
    )
    
    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    
    # 返回结果字典
    result = {
        'model_path': model_path,
        'epoch': epoch,
        'acc': acc,
        'auc': auc,
        'f1': macro_f1,  # 这里的 f1 代表 macro F1
        'precision': precision,
        'recall': recall,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP': TP,
        'cm': cm,
        'report': report,
        'time': elapsed,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }
    
    print(f"=> Test finished. Time: {int(m)}m {s:.1f}s")
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    f1_str = f"{macro_f1:.4f}"
    print(f"=> Accuracy: {acc:.4f}, AUC: {auc_str}, Macro F1: {f1_str}")
    
    return result


def save_classification_result(result, save_dir, model_file):
    """保存单个模型的分类结果"""
    epoch = result['epoch']
    test_save_dir = os.path.join(save_dir, 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    result_file = os.path.join(test_save_dir, f"{os.path.splitext(model_file)[0]}.txt")
    
    with open(result_file, "w") as f:
        f.write(f"Model: {result['model_path']}\n")
        f.write(f"Epoch: {epoch}\n\n")
        f.write(f"Overall accuracy: {result['acc']:.4f}\n\n")
        
        if result['auc'] is not None:
            f.write("Confusion Matrix Details:\n")
            f.write(f"TN (True Negative, 良性→良性): {result['TN']}\n")
            f.write(f"FP (False Positive, 良性→恶性): {result['FP']}\n")
            f.write(f"FN (False Negative, 恶性→良性): {result['FN']}\n")
            f.write(f"TP (True Positive, 恶性→恶性): {result['TP']}\n\n")
            f.write(f"AUC (Area Under ROC Curve): {result['auc']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"Macro F1 Score: {result['f1']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(result['cm']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(result['report'] + "\n")
    
    return result_file


def append_summary_result(result, summary_filename, is_first=False):
    """追加单个模型的汇总结果到文件"""
    # 如果是第一次写入，先写入表头
    if is_first:
        with open(summary_filename, 'w') as f:
            header = "Epoch\tTN\tFP\tFN\tTP\tPrecision\tRecall\tACC\tAUC\tMacro_F1\n"
            f.write(header)
    
    # 追加当前模型的结果
    with open(summary_filename, 'a') as f:
        epoch = result['epoch']
        acc = result['acc']
        auc = result['auc'] if result['auc'] is not None else -1
        f1 = result['f1'] if result['f1'] is not None else -1
        precision = result['precision'] if result['precision'] is not None else -1
        recall = result['recall'] if result['recall'] is not None else -1
        TN = result['TN'] if result['TN'] is not None else -1
        FP = result['FP'] if result['FP'] is not None else -1
        FN = result['FN'] if result['FN'] is not None else -1
        TP = result['TP'] if result['TP'] is not None else -1
        
        # 格式化输出 - 先判断值再格式化
        auc_str = f"{auc:.4f}" if auc >= 0 else "N/A"
        f1_str = f"{f1:.4f}" if f1 >= 0 else "N/A"
        precision_str = f"{precision:.4f}" if precision >= 0 else "N/A"
        recall_str = f"{recall:.4f}" if recall >= 0 else "N/A"
        TN_str = str(TN) if TN >= 0 else "N/A"
        FP_str = str(FP) if FP >= 0 else "N/A"
        FN_str = str(FN) if FN >= 0 else "N/A"
        TP_str = str(TP) if TP >= 0 else "N/A"
        
        # 按照新顺序：Epoch、TN、FP、FN、TP、Precision、Recall、Accuracy、AUC、F1
        line = f"{epoch}\t{TN_str}\t{FP_str}\t{FN_str}\t{TP_str}\t{precision_str}\t{recall_str}\t{acc:.4f}\t{auc_str}\t{f1_str}\n"
        f.write(line)


def sort_summary_by_f1(summary_filename):
    """对汇总文件按 Macro_F1 从大到小排序"""
    if not os.path.exists(summary_filename):
        print(f"Warning: Summary file {summary_filename} not found, skipping sort.")
        return
    
    # 读取文件内容
    with open(summary_filename, 'r') as f:
        lines = f.readlines()
    
    if len(lines) <= 1:
        print(f"Warning: Summary file {summary_filename} has no data rows, skipping sort.")
        return
    
    # 分离表头和数据行
    header = lines[0]
    data_lines = lines[1:]
    
    # 定义排序函数：提取 Macro_F1（最后一列）进行排序
    def get_f1_value(line):
        parts = line.strip().split('\t')
        if len(parts) < 10:
            return -1.0  # 如果列数不对，返回最小值
        f1_str = parts[9]  # Macro_F1 是第10列（索引9）
        try:
            return float(f1_str)
        except ValueError:
            # 如果是 "N/A" 或其他非数字，返回最小值
            return -1.0
    
    # 按 F1 从大到小排序
    sorted_data_lines = sorted(data_lines, key=get_f1_value, reverse=True)
    
    # 写回文件
    with open(summary_filename, 'w') as f:
        f.write(header)  # 先写表头
        f.writelines(sorted_data_lines)  # 再写排序后的数据
    
    print(f"=> Summary file sorted by Macro_F1 (descending order)")


def main():
    # ===== 1. 读取配置 =====
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    opt.print_options()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_dir = opt.test['img_dir']
    color_label_dir = opt.test['mask_color']
    save_dir = opt.test['save_dir']
    models_dir = opt.test['model_path']  # 指向模型文件夹
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # ===== 2. 定义数据增强 =====
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # ===== 3. 构建测试数据集 =====
    print("=> Building classification test dataset...")
    test_dataset = ClassificationDataFolder(
        img_dir=img_dir,
        label_dir=color_label_dir,
        data_transform=test_transform,
        target_size=opt.train['input_size'][0]
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.train['batch_size'],
        shuffle=False,
        num_workers=opt.train['workers'],
        drop_last=False
    )
    
    # ===== 4. 获取模型文件夹中的所有模型文件 =====
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth.tar') or f.endswith('.pth')]
    print(f"\nFound {len(model_files)} model files in directory: {models_dir}")
    
    if not model_files:
        print("No model files found in the specified directory. Exiting.")
        return
    
    # 获取图像目录的基本名称（用于结果文件名）
    img_dir_basename = os.path.basename(os.path.normpath(img_dir))
    
    # 定义汇总文件名（不再单独保存 best_model_info 文件）
    summary_filename = os.path.join(save_dir, f'{img_dir_basename}_all_models_summary.txt')
    
    # 初始化存储最佳结果的变量
    best_auc = -1.0
    best_acc = -1.0
    best_f1 = -1.0
    best_model_info = {}
    
    # 定义超时时间（25分钟，转换为秒）
    timeout_seconds = 25 * 60
    timeout_log_filename = os.path.join(save_dir, f'{img_dir_basename}_timeout_models.txt')
    
    all_results = []
    
    # ===== 5. 遍历所有模型文件进行测试 =====
    total_models = len(model_files)
    overall_start_time = time.time()
    
    for idx, model_file in enumerate(model_files):
        model_path = os.path.join(models_dir, model_file)
        print(f"\n{'='*90}")
        print(f"Processing model [{idx+1}/{total_models}]: {model_file}")
        
        model_start_time = time.time()
        
        # 测试单个模型
        result = test_single_model(model_path, test_loader, opt, device, idx+1, total_models)
        
        if result is None:
            print(f"Failed to test model {model_file}, skipping...")
            continue
        
        # 检查是否超时
        model_elapsed = time.time() - model_start_time
        if model_elapsed > timeout_seconds:
            print(f"Model test timeout (took {model_elapsed/60:.2f} minutes), recording...")
            with open(timeout_log_filename, 'a') as timeout_log:
                timeout_log.write(f"{model_file},{model_elapsed/60:.2f}分钟\n")
            continue
        
        # 保存当前模型结果
        result_file = save_classification_result(result, save_dir, model_file)
        result['result_file'] = result_file
        result['model_file'] = model_file  # 添加模型文件名到结果中
        all_results.append(result)
        
        # 立即写入汇总文件（第一次写入时包含表头）
        is_first = (idx == 0)
        append_summary_result(result, summary_filename, is_first=is_first)
        
        # 打印当前模型结果
        # print(f"\nModel Results:")
        # print(f"  Accuracy: {result['acc']:.4f}")
        # if result['auc'] is not None:
        #     print(f"  AUC: {result['auc']:.4f}")
        #     print(f"  F1: {result['f1']:.4f}")
        #     print(f"  Precision: {result['precision']:.4f}")
        #     print(f"  Recall: {result['recall']:.4f}")
        
        # 检查是否为当前最佳模型（基于F1）
        if result['f1'] is not None and result['f1'] > best_f1:
            best_f1 = result['f1']
            best_auc = result['auc']
            best_acc = result['acc']
            best_model_info = {
                'model_file': model_file,
                'model_path': model_path,
                'epoch': result['epoch'],
                'acc': best_acc,
                'auc': best_auc,
                'f1': best_f1,
                'precision': result['precision'],
                'recall': result['recall'],
                'TN': result['TN'],
                'FP': result['FP'],
                'FN': result['FN'],
                'TP': result['TP'],
                'result_file': result_file
            }
            print(f"\n*** New best model found! F1: {best_f1:.4f}, AUC: {best_auc:.4f}, Acc: {best_acc:.4f} ***")
    
    # ===== 6. 所有模型测试完成后，打印最佳模型信息 =====
    overall_elapsed = time.time() - overall_start_time
    overall_m, overall_s = divmod(overall_elapsed, 60)
    
    if best_f1 > -1:
        print("\n" + "="*90)
        print("Best Model Information:")
        print(f"Model File: {best_model_info['model_file']}")
        print(f"Epoch: {best_model_info['epoch']}")
        print(f"Accuracy: {best_model_info['acc']:.4f}")
        print(f"AUC: {best_model_info['auc']:.4f}")
        print(f"F1 Score: {best_model_info['f1']:.4f}")
        print(f"Precision: {best_model_info['precision']:.4f}")
        print(f"Recall: {best_model_info['recall']:.4f}")
        print(f"TN: {best_model_info['TN']}, FP: {best_model_info['FP']}, "
              f"FN: {best_model_info['FN']}, TP: {best_model_info['TP']}")
        print(f"Result File: {best_model_info['result_file']}")
        print("="*90)
    else:
        print("\nNo valid models were tested.")
    
    # ===== 7. 对所有模型结果按 Macro_F1 排序 =====
    sort_summary_by_f1(summary_filename)
    
    print(f"\n=> All models tested. Total time: {int(overall_m)}m {overall_s:.1f}s")
    print(f"=> Results saved to: {save_dir}")
    print(f"=> Summary file: {summary_filename} (sorted by Macro_F1)")


if __name__ == '__main__':
    main()