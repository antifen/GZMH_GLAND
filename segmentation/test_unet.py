import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from skimage import measure, io
import cv2
from model.segmentation.DeepLabv3_plus import DeepLabv3_plus

from model.segmentation.FullNet import Unet_18 as Unet       #导入unet模型
# from FullNet import Unet_50 as Unet

from scipy.ndimage import gaussian_filter1d
import utils_unet as utils
import time
from datetime import datetime

from options import Options
from my_transforms import get_transforms

def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    opt.print_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 获取设备
    
    img_dir = opt.test['img_dir']
    label_dir = opt.test['label_dir']
    mask_color_dir = opt.test['mask_color']
    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    tta = opt.test['tta']
    hausdorff_flag = opt.test['hausdorff']

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # data transforms
    test_transform = get_transforms(opt.transform['test'])

    # load model
    model = Unet(3, 3)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    best_checkpoint = torch.load(model_path,weights_only=False)  #5080专用
    model.load_state_dict(best_checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))
    model = model.module
    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    # TP, FP, FN, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, seg_area
    accumulated_metrics = np.zeros(11)
    all_results = dict()

     # 添加开始时间记录
    start_time = time.time()
    start_time_1 = datetime.now().strftime('%H:%M')
    print("测试开始时间：" + start_time_1 + "\n")

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)
    
    for img_name in img_names:
        # 记录单张图片开始处理时间
        img_start_time = time.time()
        
        # 初始化步骤计时器
        step_times = {
            "prob_map": 0.0,
            "postprocess": 0.0,
            "metrics": 0.0,
            "save": 0.0
        }
        
        # load test image
        print('=> Processing image {:s}'.format(img_name))
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

        # ==================== 步骤1: 计算概率图 ====================
        step1_start = time.time()
        input = test_transform((img,))[0].unsqueeze(0).cuda()

        print('\tComputing output probability maps...')
        prob_maps = get_probmaps(input, model, opt)
        if tta:
            tta_start = time.time()
            img_hf = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_vf = img.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_hf = test_transform((img_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_vf = test_transform((img_vf,))[0].unsqueeze(0)  # vertical flip input
            input_hvf = test_transform((img_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_hf = get_probmaps(input_hf, model, opt)
            prob_maps_vf = get_probmaps(input_vf, model, opt)
            prob_maps_hvf = get_probmaps(input_hvf, model, opt)

            # re flip
            prob_maps_hf = np.flip(prob_maps_hf, 2)
            prob_maps_vf = np.flip(prob_maps_vf, 1)
            prob_maps_hvf = np.flip(np.flip(prob_maps_hvf, 1), 2)

            # rotation 90 and flips
            img_r90 = img.rotate(90, expand=True)
            img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_r90 = test_transform((img_r90,))[0].unsqueeze(0)
            input_r90_hf = test_transform((img_r90_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_r90_vf = test_transform((img_r90_vf,))[0].unsqueeze(0)  # vertical flip input
            input_r90_hvf = test_transform((img_r90_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_r90 = get_probmaps(input_r90, model, opt)
            prob_maps_r90_hf = get_probmaps(input_r90_hf, model, opt)
            prob_maps_r90_vf = get_probmaps(input_r90_vf, model, opt)
            prob_maps_r90_hvf = get_probmaps(input_r90_hvf, model, opt)

            # re flip
            prob_maps_r90 = np.rot90(prob_maps_r90, k=3, axes=(1, 2))
            prob_maps_r90_hf = np.rot90(np.flip(prob_maps_r90_hf, 2), k=3, axes=(1, 2))
            prob_maps_r90_vf = np.rot90(np.flip(prob_maps_r90_vf, 1), k=3, axes=(1, 2))
            prob_maps_r90_hvf = np.rot90(np.flip(np.flip(prob_maps_r90_hvf, 1), 2), k=3, axes=(1, 2))

            prob_maps = (prob_maps + prob_maps_hf + prob_maps_vf + prob_maps_hvf
                         + prob_maps_r90 + prob_maps_r90_hf + prob_maps_r90_vf + prob_maps_r90_hvf) / 8
            step_times["tta"] = time.time() - tta_start
        
        step_times["prob_map"] = time.time() - step1_start
        # print(f'\tProbability maps computed in {step_times["prob_map"]:.2f}s')
        # if tta:
        #     print(f'\tTTA took {step_times["tta"]:.2f}s')

        # ==================== 步骤2: 后处理 ====================
        step2_start = time.time()
        pred = np.argmax(prob_maps, axis=0)  # 原始预测
        pred_inside = pred == 1   # 腺体内部区域
        pred2 = morph.remove_small_objects(pred_inside, opt.post['min_area'])  # remove small object

        if 'scale' in opt.transform['test']:
            pred2 = Image.fromarray(pred2.astype(np.uint8) * 255).resize((ori_w, ori_h), resample=Image.BILINEAR)
            pred2 = np.array(pred2)
            pred2 = (pred2 > 127.5)

        pred_labeled = measure.label(pred2)   # connected component labeling
        pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(opt.post['radius']))
        # pred_labeled = morph.dilation(pred_labeled, footprint=morph.disk(opt.post['radius']))  #5080专用


        step_times["postprocess"] = time.time() - step2_start
        # print(f'\tPost-processing completed in {step_times["postprocess"]:.2f}s')

        # ==================== 步骤3: 指标计算 ====================
        metrics_time = 0.0
        if eval_flag:
            metrics_start = time.time()
            print('\tComputing metrics...')
            result = utils.accuracy_pixel_level(np.expand_dims(pred_labeled>0,0), np.expand_dims(label_img>0, 0))
            pixel_accu = result[0]

            single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img,hausdorff_flag)
            accumulated_metrics += utils.gland_accuracy_object_level_all_images(pred_labeled, label_img,hausdorff_flag)

            all_results[name] = tuple([pixel_accu, *single_image_result])
            metrics_time = time.time() - metrics_start
            step_times["metrics"] = metrics_time
            # print(f'\tMetrics computed in {metrics_time:.2f}s')

        # ==================== 步骤4: 结果保存 ====================
        save_time = 0.0
        if save_flag:
            save_start = time.time()
            print('\tSaving image results...')
            io.imsave('{:s}/{:s}_prob_inside.png'.format(prob_maps_folder, name), (prob_maps[1,:,:] * 255).astype(np.uint8))
            io.imsave('{:s}/{:s}_prob_contour.png'.format(prob_maps_folder, name), (prob_maps[2,:,:] * 255).astype(np.uint8))
            final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
            # final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

            # save colored objects
            pred_colored = np.zeros((ori_h, ori_w, 3))
            for k in range(1, pred_labeled.max() + 1):
                pred_colored[pred_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            # io.imsave(filename, (pred_colored * 255).astype(np.uint8))
            # 添加可视化拼接：将预测结果和标签图拼接在一起
            if eval_flag:
                 # 加载原图
                original_img_path = f'{img_dir}/{img_name}'
                if os.path.exists(original_img_path):
                    original_img = io.imread(original_img_path)
                    # 确保所有图像尺寸相同
                    if original_img.shape[:2] == pred_colored.shape[:2]:
                        # 加载彩色标签图
                        color_label_path = f'{mask_color_dir}/{name}_label.png'  #彩色标签图的路径
                        if os.path.exists(color_label_path):
                            color_label_img = io.imread(color_label_path)
                            # 确保两个图像尺寸相同
                            if color_label_img.shape[:2] == pred_colored.shape[:2]:
                                # 创建白色间隔条（宽度为10像素）
                                height = pred_colored.shape[0]
                                white_space = np.ones((height, 10, 3))  # 白色间隔
                                
                                # 水平拼接标签图、间隔和预测结果（标签图在左，测试图在右）
                                combined_img = np.hstack((original_img / 255.0,white_space, color_label_img / 255.0, white_space, pred_colored))
                                combined_filename = '{:s}/{:s}_comparison.png'.format(seg_folder, name)
                                io.imsave(combined_filename, (combined_img * 255).astype(np.uint8))
                                print(f'\tSaved comparison image: {combined_filename}')
                else:
                    print("图片不存在")

            save_time = time.time() - save_start
            step_times["save"] = save_time

        # 计算并输出单张图片处理耗时
        img_elapsed = time.time() - img_start_time
        minutes, seconds = divmod(img_elapsed, 60)
        print(f'\tImage {img_name} processed in {int(minutes)}m {seconds:.1f}s')
        print(f'\tStep times: Prob Map: {step_times["prob_map"]:.1f}s, Post-process: {step_times["postprocess"]:.1f}s, Metrics: {step_times["metrics"]:.1f}s, Save: {step_times["save"]:.1f}s\n')

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images\n'.format(counter))

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

    print('=> Processed all {:d} images'.format(counter))
    if eval_flag:
        print('Average of all images:\n'
              'recall: {r[1]:.4f}\n'
              'precision: {r[2]:.4f}\n'
              'F1: {r[3]:.4f}\n'
              'dice: {r[4]:.4f}\n'
              'iou: {r[5]:.4f}\n'
              'haus: {r[6]:.4f}'.format(r=avg_results))

        strs = img_dir.split('/')
        header = ['pixel_acc','recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff']
        save_results(header, avg_results, all_results,
                     '{:s}/{:s}_result.txt'.format(save_dir, strs[-1]))

    # 添加结束时间计算和输出
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 转换时间为分秒格式
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"=> 测试时间为：{minutes:.0f}m {seconds:.2f}s")


def smooth_segmentation(label_mask, keep_freq=20):
    """
    使用傅里叶描述子平滑腺体分割边缘，去除长波长边界起伏。
    参数 keep_freq 控制保留的低频量，值越小越光滑。
    """
    label_mask = label_mask.astype(np.uint8)
    smoothed_mask = np.zeros_like(label_mask)
    num_labels = label_mask.max()

    for i in range(1, num_labels + 1):
        obj_mask = (label_mask == i).astype(np.uint8)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if len(cnt) < 10:
                continue  # 小对象跳过

            cnt = cnt[:, 0, :]  # shape (N, 2)
            complex_coords = cnt[:, 0] + 1j * cnt[:, 1]

            # 傅里叶变换
            fft_result = np.fft.fft(complex_coords)
            # 低通滤波（只保留前 keep_freq 个频率）
            fft_result[keep_freq:-keep_freq] = 0
            # 逆变换
            smoothed_coords = np.fft.ifft(fft_result)

            smoothed_cnt = np.stack((smoothed_coords.real, smoothed_coords.imag), axis=1)
            smoothed_cnt = np.round(smoothed_cnt).astype(np.int32)
            smoothed_cnt = smoothed_cnt.reshape(-1, 1, 2)

            # 画出平滑后的轮廓
            cv2.drawContours(smoothed_mask, [smoothed_cnt], -1, int(i), thickness=-1)

    return smoothed_mask

def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    # 确保input在GPU上
    input = input.cuda()  # 添加此行

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