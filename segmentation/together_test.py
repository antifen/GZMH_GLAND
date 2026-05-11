import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from skimage import measure, io

from model.segmentation.DeepLabv3_plus import DeepLabv3_plus

import utils as utils
import time
from datetime import datetime

from options import Options
from my_transforms import get_transforms


def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.print_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_dir = opt.test['img_dir']
    label_dir = opt.test['label_dir']
    save_dir = opt.test['save_dir']
    models_dir = opt.test['model_path']  # 指向模型文件夹
    tta = opt.test['tta']
    hausdorff_flag = opt.test['hausdorff']

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 检查是否需要计算精度
    eval_flag = True if label_dir else False

    # 数据转换
    test_transform = get_transforms(opt.transform['test'])

    # 获取模型文件夹中的所有模型文件
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth.tar') or f.endswith('.pth')]
    print(f"Found {len(model_files)} model files in directory: {models_dir}")
    
    if not model_files:
        print("No model files found in the specified directory. Exiting.")
        return
    
    # 获取图像目录的基本名称（用于结果文件名）
    img_dir_basename = os.path.basename(os.path.normpath(img_dir))

    # 定义最佳模型信息文件名
    best_info_filename = os.path.join(save_dir, f'{img_dir_basename}_best_model_info.txt')

    # 只保存一份选项文件的路径
    global_option_filename = os.path.join(save_dir, f'{img_dir_basename}_options.txt')
    option_saved = False

    # 初始化存储最佳结果的变量
    best_f1 = -1.0
    best_model_info = {}

    # 获取模型总数
    total_models = len(model_files)
    
    # 定义超时时间（25分钟，转换为秒）
    timeout_seconds = 25 * 60
    # 定义超时记录文件名
    timeout_log_filename = os.path.join(save_dir, f'{img_dir_basename}_timeout_models.txt')
    
    # 遍历模型文件夹中的每个模型文件
    for idx, model_file in enumerate(model_files):
        model_path = os.path.join(models_dir, model_file)
        print(f"\n{'='*90}")
        print(f"Processing model [{idx+1}/{total_models}]: {model_file}")
        
        # 初始化模型
        model = DeepLabv3_plus(
            nInputChannels=opt.model['in_c'],
            n_classes=opt.model['out_c'],
            os=opt.model['os'],
            resnet_type=opt.model['resnet_type'],
            _print=False
        )
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

        # 加载当前模型
        print(f"=> loading trained model: {model_file}")
        try:
            best_checkpoint = torch.load(model_path, map_location=device,weights_only=False)
            model.load_state_dict(best_checkpoint['state_dict'])
            epoch = best_checkpoint.get('epoch', 'unknown')
            print(f"=> loaded model at epoch {epoch}")
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
            continue
            
        model = model.module
        model.eval()  # 切换到评估模式

        counter = 0
        print("=> Test begins:")

        img_names = os.listdir(img_dir)

        # TP, FP, FN, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, seg_area
        accumulated_metrics = np.zeros(11)
        all_results = dict()

        # 添加开始时间记录
        start_time = time.time()
        start_time_1 = datetime.now().strftime('%H:%M')
        print("本轮开始时间：" + start_time_1)

        # 初始化超时标志
        timeout_occurred = False
        last_processed_image = None  # 记录最后处理的图片名称

        for img_name in img_names:
            # 加载测试图像
            img_path = '{:s}/{:s}'.format(img_dir, img_name)
            img = Image.open(img_path)
            ori_h = img.size[1]
            ori_w = img.size[0]
            name = os.path.splitext(img_name)[0]
            if eval_flag:
                # 根据数据集类型生成标签文件路径
                if opt.dataset == 'CRAG':
                    label_path = '{:s}/{:s}.png'.format(label_dir, name)
                else:  # 默认为CRAG格式
                    label_path = '{:s}/{:s}_mask.png'.format(label_dir, name)
                label_img = io.imread(label_path)

            input = test_transform((img,))[0].unsqueeze(0).cuda()

            prob_maps = get_probmaps(input, model, opt)
            if tta:
                # 测试时增强处理
                img_hf = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_vf = img.transpose(Image.FLIP_TOP_BOTTOM)
                img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)

                input_hf = test_transform((img_hf,))[0].unsqueeze(0).cuda()
                input_vf = test_transform((img_vf,))[0].unsqueeze(0).cuda()
                input_hvf = test_transform((img_hvf,))[0].unsqueeze(0).cuda()

                prob_maps_hf = get_probmaps(input_hf, model, opt)
                prob_maps_vf = get_probmaps(input_vf, model, opt)
                prob_maps_hvf = get_probmaps(input_hvf, model, opt)

                prob_maps_hf = np.flip(prob_maps_hf, 2)
                prob_maps_vf = np.flip(prob_maps_vf, 1)
                prob_maps_hvf = np.flip(np.flip(prob_maps_hvf, 1), 2)

                img_r90 = img.rotate(90, expand=True)
                img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)
                img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)
                img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)

                input_r90 = test_transform((img_r90,))[0].unsqueeze(0).cuda()
                input_r90_hf = test_transform((img_r90_hf,))[0].unsqueeze(0).cuda()
                input_r90_vf = test_transform((img_r90_vf,))[0].unsqueeze(0).cuda()
                input_r90_hvf = test_transform((img_r90_hvf,))[0].unsqueeze(0).cuda()

                prob_maps_r90 = get_probmaps(input_r90, model, opt)
                prob_maps_r90_hf = get_probmaps(input_r90_hf, model, opt)
                prob_maps_r90_vf = get_probmaps(input_r90_vf, model, opt)
                prob_maps_r90_hvf = get_probmaps(input_r90_hvf, model, opt)

                prob_maps_r90 = np.rot90(prob_maps_r90, k=3, axes=(1, 2))
                prob_maps_r90_hf = np.rot90(np.flip(prob_maps_r90_hf, 2), k=3, axes=(1, 2))
                prob_maps_r90_vf = np.rot90(np.flip(prob_maps_r90_vf, 1), k=3, axes=(1, 2))
                prob_maps_r90_hvf = np.rot90(np.flip(np.flip(prob_maps_r90_hvf, 1), 2), k=3, axes=(1, 2))

                prob_maps = (prob_maps + prob_maps_hf + prob_maps_vf + prob_maps_hvf
                             + prob_maps_r90 + prob_maps_r90_hf + prob_maps_r90_vf + prob_maps_r90_hvf) / 8

            pred = np.argmax(prob_maps, axis=0)
            pred_inside = pred == 1
            pred2 = morph.remove_small_objects(pred_inside, opt.post['min_area'])

            if 'scale' in opt.transform['test']:
                pred2 = Image.fromarray(pred2.astype(np.uint8) * 255).resize((ori_w, ori_h), resample=Image.BILINEAR)
                pred2 = np.array(pred2)
                pred2 = (pred2 > 127.5)

            pred_labeled = measure.label(pred2)
            pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(opt.post['radius']))
            # pred_labeled = morph.dilation(pred_labeled, footprint=morph.disk(opt.post['radius']))  #5080专用

            if eval_flag:
                result = utils.accuracy_pixel_level(np.expand_dims(pred_labeled>0,0), np.expand_dims(label_img>0, 0))
                pixel_accu = result[0]

                single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img,hausdorff_flag)
                accumulated_metrics += utils.gland_accuracy_object_level_all_images(pred_labeled, label_img,hausdorff_flag)

                all_results[name] = tuple([pixel_accu, *single_image_result])

            counter += 1
            if counter % 10 == 0:
                print(f'\tProcessed {counter} images')
                
            # 记录最后成功处理的图片名称
            last_processed_image = img_name
            
            # 检查是否超时 - 放在处理完图片后
            current_time = time.time()
            e_time = current_time - start_time
            if e_time > timeout_seconds:
                print(f"模型测试超时（已运行 {e_time/60:.2f} 分钟），停止测试。最后处理的图片: {last_processed_image}")
                # 记录超时模型文件名和最后处理的图片名称
                with open(timeout_log_filename, 'a') as timeout_log:
                    timeout_log.write(f"{model_file},{last_processed_image},{e_time/60:.2f}分钟\n")
                timeout_occurred = True
                break  # 停止当前模型的测试

        # 如果发生超时，跳过该模型的结果保存和比较
        if timeout_occurred:
            print("模型因超时未完成测试，跳过结果保存和比较。")
            continue  # 继续处理下一个模型
            
        if eval_flag and counter > 0:
            # 计算总体指标
            TP, FP, FN, dice_g, dice_s, iou_g, iou_s, hausdorff_g, hausdorff_s, \
            gt_objs_area, pred_objs_area = accumulated_metrics

            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            F1 = 2 * TP / (2 * TP + FP + FN)
            dice = (dice_g / gt_objs_area + dice_s / pred_objs_area) / 2
            iou = (iou_g / gt_objs_area + iou_s / pred_objs_area) / 2
            haus = (hausdorff_g / gt_objs_area + hausdorff_s / pred_objs_area) / 2

            avg_pixel_accu = -1
            avg_results = [avg_pixel_accu, recall, precision, F1, dice, iou, haus]

            print(f'=> Processed all {counter} images')
            print('Average of all images:\n'
                  'recall: {r[1]:.4f}\n'
                  'precision: {r[2]:.4f}\n'
                  'F1: {r[3]:.4f}\n'
                  'dice: {r[4]:.4f}\n'
                  'iou: {r[5]:.4f}\n'
                  'haus: {r[6]:.4f}'.format(r=avg_results))

            # 保存当前模型的F1值
            current_f1 = F1

            result_dir = os.path.join(save_dir, img_dir_basename)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # 保存测试结果文件（文件名包含epoch）
            result_filename = '{:s}/{:s}/{:s}_epoch_{:d}_result.txt'.format(
                save_dir, img_dir_basename,img_dir_basename, int(epoch))
                
            header = ['pixel_acc','recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff']
            save_results(header, avg_results, all_results, result_filename)
            
            # 保存选项文件（只保存一次）
            if not option_saved:
                save_options(opt, global_option_filename)
                option_saved = True

             # 检查是否为当前最佳模型
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_info = {
                    'model_file': model_file,
                    'epoch': epoch,
                    'f1': best_f1,
                    'result_file': result_filename,
                    'option_file': global_option_filename
                }
                print(f"\nNew best model found! F1: {best_f1:.4f}")

                # 立即保存最佳模型信息到文件
                with open(best_info_filename, 'w') as f:
                    f.write("Current Best Model Information:\n")
                    f.write(f"Model File: {best_model_info['model_file']}\n")
                    f.write(f"Epoch: {best_model_info['epoch']}\n")
                    f.write(f"F1 Score: {best_model_info['f1']:.4f}\n")
                    f.write(f"Result File: {best_model_info['result_file']}\n")
                    f.write(f"Option File: {best_model_info['option_file']}\n")
                    # 添加测试状态（进行中）
                    f.write(f"Status: Testing in progress... ({idx+1}/{total_models} models processed)\n")

            # 添加结束时间计算和输出
            end_time = time.time()
            elapsed_time = end_time - start_time
            # 转换时间为分秒格式
            minutes, seconds = divmod(elapsed_time, 60)
            print(f"=> This round of testing took {minutes:.0f}m {seconds:.2f}s")

    # 所有模型测试完成后，打印最佳模型信息
    if best_f1 > -1:
        print("\n" + "="*90)
        print("Best Model Information:")
        print(f"Model File: {best_model_info['model_file']}")
        print(f"Epoch: {best_model_info['epoch']}")
        print(f"F1 Score: {best_model_info['f1']:.4f}")
        print(f"Result File: {best_model_info['result_file']}")
        print(f"Option File: {best_model_info['option_file']}")
        print("="*90 + "\n")

        # 更新最佳模型信息文件状态为完成
        with open(best_info_filename, 'r+') as f:
            content = f.readlines()
            # 移除原来的状态行
            content = content[:-1] if content else []
            f.seek(0)
            f.truncate()
            f.writelines(content)
            # 添加完成状态
            f.write(f"Status: Testing completed. ({total_models}/{total_models} models processed)\n")
    else:
        print("\nNo valid models were tested.")

def save_options(opt, filename):
    """保存选项到文件（保持原始格式）"""
    with open(filename, 'w') as f:
        # 保存所有选项组
        f.write("[test]\n")
        for key, value in opt.test.items():
            f.write(f"{key} = {value}\n")

        f.write("\n[post]\n")
        for key, value in opt.post.items():
            f.write(f"{key} = {value}\n")

        f.write("\n[model]\n")
        for key, value in opt.model.items():
            f.write(f"{key} = {value}\n")

def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


def save_results(header, avg_results, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_results[i]))
        file.write('{:.4f}\n'.format(avg_results[N - 1]))
        file.write('\n')

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


if __name__ == '__main__':
    main()