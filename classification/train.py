import faulthandler
faulthandler.enable()
import argparse
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import shutil
import numpy as np
import logging
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score

from classification_data_folder import ClassificationDataFolder
from options import Options

# 导入分类模型
from torchvision.models import resnet50, resnet101, densenet121, densenet169, efficientnet_b0, efficientnet_b3
import timm  # 如果安装了timm库

VAL_F1 = 0.80  # 改为F1阈值

def get_model(model_name, num_classes=2, pretrained=True):
    """获取分类模型"""
    # --- ResNet 系列 ---
    if model_name == 'resnet50':
        # 如果 pretrained 为 True，使用默认的最佳权重；否则为 None
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # --- DenseNet 系列 ---
    elif model_name == 'densenet121':
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'densenet169':
        weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
        model = models.densenet169(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # --- EfficientNet 系列 ---
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet 的 classifier 是一个 Sequential，包含 Dropout 和 Linear
        # 这里我们保持原有的 Dropout 比例，只替换 Linear 层
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif model_name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        # 修改分类头 (classifier 是一个 Sequential，最后一层是 Linear)
        # ConvNeXt 的最后一层通常叫 model.classifier[2]
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'resnext50':
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        model = models.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    # 推荐 swin_t (Tiny版本，对应 ResNet50 量级)
    elif model_name == 'swin_t':
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'regnet_y_3_2gf':
        weights = models.RegNet_Y_3_2GF_Weights.DEFAULT if pretrained else None
        model = models.regnet_y_3_2gf(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vit_base':
        # torchvision 使用 weights 参数，这是更现代和推荐的方式
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        
        # 替换最后的分类头
        # vit_b_16 的分类头是 model.heads.head
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        # MobileNetV3 的分类头是 classifier (最后一层是 Linear)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def main():
    global opt, best_f1, tb_writer, logger, logger_results
    best_f1 = 0  # 改为best_f1
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpu'])
    
    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    # ----- create model ----- #
    model = get_model(
        model_name=opt.model['name'],
        num_classes=opt.model['num_classes'],
        pretrained=opt.model['pretrained']
    )
    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 修改后建议：使用余弦退火，让LR在训练过程中平滑下降
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.train['num_epochs'], eta_min=1e-6)

    # ----- define criterion ----- #
    # criterion = nn.CrossEntropyLoss().cuda()
    # 权重 = [良性权重, 恶性权重]
    # 对应 label 0 和 label 1
    class_weights = torch.tensor([1.92, 1.0]).float().cuda()
    
    # 结合 Label Smoothing 防止过拟合
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).cuda()

    # ----- data transforms ----- #
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(
            degrees=180, 
            fill=255  # <--- 重要！保持背景为白色
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 加大明暗对比变化，模拟染色深浅；减小色相变化，防止颜色失真
        # transforms.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.4, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ----- load data ----- #
    train_dataset = ClassificationDataFolder(
        img_dir='{:s}/train'.format(opt.train['img_dir']),
        label_dir='{:s}/train'.format(opt.train['label_dir']),
        data_transform=train_transform,
        target_size=opt.train['input_size'][0]
    )
    
    val_dataset = ClassificationDataFolder(
        img_dir='{:s}/test'.format(opt.train['img_dir']),
        label_dir='{:s}/test'.format(opt.train['label_dir']),
        data_transform=val_transform,
        target_size=opt.train['input_size'][0]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.train['batch_size'], shuffle=False,
                            num_workers=opt.train['workers'], drop_last=False)

    # ----- optionally load from a checkpoint ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'], weights_only=False)
            opt.train['start_epoch'] = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    # ----- training and validation ----- #
    for epoch in range(opt.train['start_epoch'], opt.train['num_epochs']):
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, opt.train['num_epochs']))
        train_results = train(train_loader, model, optimizer, criterion, epoch)
        train_loss, train_acc = train_results

        # evaluate on validation set
        with torch.no_grad():
            val_loss, val_acc, val_f1, val_tn, val_fp, val_fn, val_tp = validate(val_loader, model, criterion)

        scheduler.step()

        is_best = val_f1 > best_f1  # 改为基于F1判断
        best_f1 = max(val_f1, best_f1)  # 更新best_f1

        if val_f1 >= VAL_F1:  # 改为基于F1阈值
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,  # 保存best_f1
                'best_acc': val_acc,  # 同时保存当前acc用于兼容
                'optimizer': optimizer.state_dict(),
            }, epoch, is_best, opt.train['save_dir'], val_f1)  # 传入val_f1

        # save the training results
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.4f}'.format(epoch+1, train_loss, train_acc, val_loss, val_acc, int(val_tn), int(val_fp), int(val_fn), int(val_tp), val_f1))
        
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        tb_writer.add_scalars('epoch_accuracies',
                              {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
        tb_writer.add_scalar('val_f1', val_f1, epoch)  # 添加F1的tensorboard记录
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    
    tb_writer.close()

def train(train_loader, model, optimizer, criterion, epoch):
    results = AverageMeter(2)
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy
        pred = output.data.max(1)[1]
        acc = pred.eq(target_var.data).float().mean()

        result = [loss.item(), acc.item()]
        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]\tLoss {r[0]:.4f}\tAcc {r[1]:.4f}'
                        .format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}\tAcc {r[1]:.4f}'.format(r=results.avg))
    return results.avg

def validate(val_loader, model, criterion):
    results = AverageMeter(2)
    model.eval()
    
    all_preds = []
    all_labels = []

    for i, (input, target) in enumerate(val_loader):
        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy
        pred = output.data.max(1)[1]
        acc = pred.eq(target_var.data).float().mean()

        results.update([loss.item(), acc.item()], input.size(0))
        
        # 收集预测和标签用于计算F1
        all_preds.append(pred.cpu().numpy())
        all_labels.append(target_var.cpu().numpy())

    # 计算F1分数（macro average）
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算混淆矩阵（如果是二分类，继续给出 TN/FP/FN/TP，方便分析）
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # 初始化变量以防混淆矩阵形状不匹配
    TN, FP, FN, TP = 0, 0, 0, 0

    if cm.shape == (2, 2):
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
    else:
        logger.warning("Confusion matrix shape is not (2, 2), cannot compute TN, FP, FN, TP.")

    # 使用 sklearn 直接计算 macro F1（支持二分类和多分类）
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # 日志输出：使用 macro F1
    logger.info('\t=> Val Avg: Loss {r[0]:.4f}\tAcc {r[1]:.4f}\tF1_macro {f1:.4f}'
                '\tTN {TN:d}\tFP {FP:d}\tFN {FN:d}\tTP {TP:d}'.format(
                    r=results.avg, f1=f1_macro, TN=TN, FP=FP, FN=FN, TP=TP))

    return results.avg[0], results.avg[1], f1_macro, TN, FP, FN, TP

def save_checkpoint(state, epoch, is_best, save_dir, val_f1):  # 参数改为val_f1
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)

    if val_f1 >= VAL_F1:  # 改为基于F1阈值
        high_f1_filename = '{:s}/checkpoint_epoch_{:03d}_f1_{:.4f}.pth.tar'.format(cp_dir, epoch+1, val_f1)
        shutil.copyfile(filename, high_f1_filename)

    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))

def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %I:%M')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
         logger_results.info('epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tTN\tFP\tFN\tTP\tval_f1')
    return logger, logger_results

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(torch.tensor(val, device="cpu"))
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()