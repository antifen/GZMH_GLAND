import os
import sys
import xml.etree.ElementTree as ET
import openslide
import numpy as np
from PIL import Image
import imageio
import cv2
from tqdm import tqdm
import csv

# ========== 全局配置 ==========
BASE_OUTPUT_DIR = './gland'

patch_level = 1
check_fit_patch_level = 1
patch_size = 1024
target_level_res = 0.25 * (2 ** patch_level)
check_fit_level_res = 0.25 * (2 ** check_fit_patch_level)

# 前景总面积占比阈值：小于 1% 则丢弃 patch
FOREGROUND_AREA_RATIO_THRESH = 0.01

# 中心随机偏移范围（像素，level 0）：[-RANDOM_OFFSET_MAX, RANDOM_OFFSET_MAX]
RANDOM_OFFSET_MAX = 300

annotation_group_dict = {'tumor gland': 'tumor gland',
                         'Non-tumor gland': 'Non-tumor gland'}

annotation_label_dict = {'tumor gland': 1,
                         'Non-tumor gland': 2}

annotation_group_color_dict = {'tumor gland': [255, 0, 0],
                               'Non-tumor gland': [0, 255, 0]}


def check_fit_into_frame(ref_annotation_dict=None, candidate_annotation_dict=None, im_read_level=0, im_read_size=None):
    if im_read_level != 0:
        print('ERROR: im_read_level != 0')
        sys.exit()

    X_cm_ref = ref_annotation_dict['X_cm']
    Y_cm_ref = ref_annotation_dict['Y_cm']

    X_min_frame = X_cm_ref - int(im_read_size[0] / 2)
    Y_min_frame = Y_cm_ref - int(im_read_size[0] / 2)
    X_max_frame = X_min_frame + im_read_size[0]
    Y_max_frame = Y_min_frame + im_read_size[0]

    X_min_candidate = candidate_annotation_dict['X_min']
    X_max_candidate = candidate_annotation_dict['X_max']
    Y_min_candidate = candidate_annotation_dict['Y_min']
    Y_max_candidate = candidate_annotation_dict['Y_max']

    if (X_min_candidate < X_min_frame) or (X_max_candidate > X_max_frame) or (Y_min_candidate < Y_min_frame) or (
            Y_max_candidate > Y_max_frame):
        return False
    else:
        return True


def get_img(slide=None, X_cm=0, Y_cm=0, im_read_level=0, im_read_size=None, res_ratio_target_to_read=1):
    if im_read_level != 0:
        print('ERROR: im_read_level != 0')
        sys.exit()

    X_min_frame = X_cm - int(im_read_size[0] / 2)
    Y_min_frame = Y_cm - int(im_read_size[1] / 2)

    im = slide.read_region((X_min_frame, Y_min_frame), im_read_level, im_read_size)
    im_arr = np.array(im)[:, :, 0:3]

    if res_ratio_target_to_read > 1:
        im_arr = np.array(im.resize((patch_size, patch_size), Image.LANCZOS))[:, :, 0:3]

    return im_arr


def get_area_by_pixel_count(X_arr, Y_arr, mask_size=None):
    if mask_size is None:
        x_min = int(np.min(X_arr)) - 1
        x_max = int(np.max(X_arr)) + 1
        y_min = int(np.min(Y_arr)) - 1
        y_max = int(np.max(Y_arr)) + 1
        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1
        X_arr_adjusted = X_arr - x_min
        Y_arr_adjusted = Y_arr - y_min
    else:
        mask_width, mask_height = mask_size
        X_arr_adjusted = X_arr
        Y_arr_adjusted = Y_arr

    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    pts = np.hstack((X_arr_adjusted[:, np.newaxis], Y_arr_adjusted[:, np.newaxis])).astype(np.int32)
    cv2.drawContours(mask, [pts], 0, 1, -1)
    pixel_count = int(np.sum(mask))
    return pixel_count


def get_mask_complete_glands(annotations_dict_list=None, center_annotation_dict=None, im_read_level=0,
                             im_read_size=None, res_ratio_target_to_read=1, center_gland_name=None):
    if im_read_level != 0:
        print('ERROR: im_read_level != 0')
        sys.exit()

    res_ratio_read_to_target = 1.0 / res_ratio_target_to_read
    mask_size = int(im_read_size[0] * res_ratio_read_to_target)

    X_cm_center = center_annotation_dict['X_cm']
    Y_cm_center = center_annotation_dict['Y_cm']

    X_offset = int(X_cm_center * res_ratio_read_to_target) - int(mask_size / 2.0)
    Y_offset = int(Y_cm_center * res_ratio_read_to_target) - int(mask_size / 2.0)

    num_instances = 1
    canvas_instance = np.zeros((mask_size, mask_size), dtype=np.uint8)
    canvas_color = np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
    canvas_binary_all = []
    label_list = list()
    area_list = list()
    center_id = 0
    center_checked = 0
    error_flag = False

    for i in range(len(annotations_dict_list)):
        temp_annotation_dict = annotations_dict_list[i]

        # 中心点对齐后质心改变，不能用坐标相等判断；用腺体 name 标识当前 patch 的中心实例
        if center_gland_name is not None:
            center_flag = temp_annotation_dict['name'] == center_gland_name
        else:
            center_flag = (temp_annotation_dict['X_cm'] == center_annotation_dict['X_cm']) and (
                        temp_annotation_dict['Y_cm'] == center_annotation_dict['Y_cm'])

        fit_into_frame_flag = check_fit_into_frame(ref_annotation_dict=center_annotation_dict,
                                                   candidate_annotation_dict=temp_annotation_dict,
                                                   im_read_level=im_read_level,
                                                   im_read_size=im_read_size)
        if not fit_into_frame_flag and not center_flag:
            continue

        canvas_binary = np.zeros((mask_size, mask_size), dtype=np.uint8)
        X_arr = np.asarray(temp_annotation_dict['X_arr'] * res_ratio_read_to_target, dtype=int) - X_offset
        Y_arr = np.asarray(temp_annotation_dict['Y_arr'] * res_ratio_read_to_target, dtype=int) - Y_offset
        annotation_group = temp_annotation_dict['annotation_group']

        pts = np.hstack((X_arr[:, np.newaxis], Y_arr[:, np.newaxis]))
        cv2.drawContours(canvas_binary, [pts], 0, (1), -1)
        if np.sum(canvas_binary) == 0:
            error_flag = True
        cv2.drawContours(canvas_instance, [pts], 0, (num_instances), -1)

        canvas_binary_all.append(canvas_binary)
        rgb_color = annotation_group_color_dict[annotation_group]
        cv2.drawContours(canvas_color, [pts], 0, rgb_color, -1)
        label_list.append(annotation_label_dict[annotation_group])

        X_arr_nos = np.asarray(temp_annotation_dict['X_arr'] * res_ratio_read_to_target, dtype=int)
        Y_arr_nos = np.asarray(temp_annotation_dict['Y_arr'] * res_ratio_read_to_target, dtype=int)
        orig_area = get_area_by_pixel_count(X_arr_nos, Y_arr_nos, mask_size=(mask_size, mask_size))
        area_list.append(orig_area)

        if center_flag:
            center_id = num_instances - 1
            if 'checked' in temp_annotation_dict['name']:
                center_checked = 1
        num_instances += 1

    if len(canvas_binary_all) == 1:
        canvas_binary_all = canvas_binary_all[0][np.newaxis, :]
    elif len(canvas_binary_all) > 1:
        canvas_binary_all = np.stack(canvas_binary_all)

    assert len(label_list) == canvas_binary_all.shape[0]
    return label_list, canvas_instance, canvas_binary_all, canvas_color, center_id, center_checked, error_flag, area_list


def Seg(slice_file, end_dir):
    global allnum

    patch_level = globals()['patch_level']
    check_fit_patch_level = globals()['check_fit_patch_level']
    patch_size = globals()['patch_size']
    target_level_res = globals()['target_level_res']
    check_fit_level_res = globals()['check_fit_level_res']
    FOREGROUND_AREA_RATIO_THRESH = globals()['FOREGROUND_AREA_RATIO_THRESH']
    RANDOM_OFFSET_MAX = globals()['RANDOM_OFFSET_MAX']

    out_dir = BASE_OUTPUT_DIR

    print(out_dir)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except Exception:
            print("An exception occurred!")

    csv_file_path = os.path.join(out_dir, 'all_slides_info.csv')
    csv_exists = os.path.isfile(csv_file_path)
    csv_header = ['wsi_id', 'number_of_patches']

    outdir_img = os.path.join(out_dir, 'img')
    outdir_label = os.path.join(out_dir, 'label')
    outdir_mask = os.path.join(out_dir, 'mask')
    outdir_binary_masks = os.path.join(out_dir, 'binary_masks')
    outdir_mask_color = os.path.join(out_dir, 'mask_color')

    for d in [outdir_img, outdir_label, outdir_mask, outdir_binary_masks, outdir_mask_color]:
        if not os.path.exists(d):
            os.makedirs(d)

    cropped_patches_filelist = os.path.join(out_dir, 'cropped_patches_filelist.txt')
    if not os.path.isfile(cropped_patches_filelist):
        with open(cropped_patches_filelist, 'a') as f_cropped_patches_filelist:
            f_cropped_patches_filelist.write('#wsi_id\tpatch_id\tX_cm\tY_cm\tcenter_id\tcenter_checked\tcenter_label\n')

    num_slides = 1

    for i in range(num_slides):
        slide_id = slice_file
        slide_path = "./data/WSI/" + slice_file + ".svs"
        slide = openslide.OpenSlide(slide_path)

        val_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
        if val_x < 0.3:
            current_res = 0.25
        elif val_x < 0.6:
            current_res = 0.5

        im_read_level = 0
        read_level_res = current_res
        res_ratio_target_to_read = target_level_res / read_level_res
        im_read_size = (int(patch_size * res_ratio_target_to_read), int(patch_size * res_ratio_target_to_read))

        res_ratio_fit_patch_level_to_read = check_fit_level_res / read_level_res
        im_read_size_fit_patch_level = (int(patch_size * res_ratio_fit_patch_level_to_read), int(patch_size * res_ratio_fit_patch_level_to_read))

        xml_file_path = "./data/ganzhou/" + slice_file + ".xml"
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        Annotations = root

        annotations_dict_list = list()
        for Annotation in Annotations:
            Annotation_PartOfGroup = Annotation.attrib['Name']
            annotation_name = Annotation.attrib['Name']
            if Annotation_PartOfGroup not in annotation_group_dict:
                continue
            annotation_group = annotation_group_dict[Annotation_PartOfGroup]

            Coordinates = Annotation.iter("Vertices")
            cnt = 0
            for Coordinate in Coordinates:
                cnt += 1
                X_list = list()
                Y_list = list()
                for Vertex in Coordinate.findall("Vertex"):
                    X_list.append(int(float(Vertex.attrib['X'])))
                    Y_list.append(int(float(Vertex.attrib['Y'])))

                X_list.append(X_list[0])
                Y_list.append(Y_list[0])

                X_arr = np.array(X_list)
                Y_arr = np.array(Y_list)

                gland_area = get_area_by_pixel_count(X_arr, Y_arr)

                X_min = np.amin(X_arr)
                X_max = np.amax(X_arr)
                Y_min = np.amin(Y_arr)
                Y_max = np.amax(Y_arr)

                X_cm = int(np.mean(X_arr))
                Y_cm = int(np.mean(Y_arr))

                temp_annotation_dict = dict()
                temp_annotation_dict['annotation_group'] = annotation_group
                temp_annotation_dict['X_arr'] = X_arr
                temp_annotation_dict['Y_arr'] = Y_arr
                temp_annotation_dict['X_min'] = X_min
                temp_annotation_dict['X_max'] = X_max
                temp_annotation_dict['Y_min'] = Y_min
                temp_annotation_dict['Y_max'] = Y_max
                temp_annotation_dict['X_cm'] = X_cm
                temp_annotation_dict['Y_cm'] = Y_cm
                temp_annotation_dict['name'] = annotation_name + "_" + str(cnt)
                temp_annotation_dict['area'] = gland_area

                annotations_dict_list.append(temp_annotation_dict)

        num_cropped_patches = 0
        large_annotation_count = 0
        pbar = tqdm(total=len(annotations_dict_list))

        processed_glands = set()

        def is_fully_contained(gland_xmin, gland_xmax, gland_ymin, gland_ymax,
                               patch_xmin, patch_xmax, patch_ymin, patch_ymax):
            return (gland_xmin >= patch_xmin and gland_xmax <= patch_xmax and
                    gland_ymin >= patch_ymin and gland_ymax <= patch_ymax)

        for i in range(len(annotations_dict_list)):
            temp_annotation_dict = annotations_dict_list[i]

            if temp_annotation_dict['name'] in processed_glands:
                pbar.update(1)
                continue

            original_X_cm = temp_annotation_dict['X_cm']
            original_Y_cm = temp_annotation_dict['Y_cm']

            # 仅随机 ±RANDOM_OFFSET_MAX 像素偏移（含端点）
            dx = np.random.randint(-RANDOM_OFFSET_MAX, RANDOM_OFFSET_MAX + 1)
            dy = np.random.randint(-RANDOM_OFFSET_MAX, RANDOM_OFFSET_MAX + 1)

            slide_width, slide_height = slide.dimensions[0], slide.dimensions[1]
            new_X_cm = original_X_cm + dx
            new_Y_cm = original_Y_cm + dy

            half_w = im_read_size[0] // 2
            half_h = im_read_size[1] // 2
            new_X_cm = max(half_w, min(new_X_cm, slide_width - half_w))
            new_Y_cm = max(half_h, min(new_Y_cm, slide_height - half_h))

            temp_center = temp_annotation_dict.copy()
            temp_center['X_cm'] = new_X_cm
            temp_center['Y_cm'] = new_Y_cm

            patch_X_min = temp_center['X_cm'] - int(im_read_size[0] / 2)
            patch_Y_min = temp_center['Y_cm'] - int(im_read_size[1] / 2)
            patch_X_max = temp_center['X_cm'] + int(im_read_size[0] / 2)
            patch_Y_max = temp_center['Y_cm'] + int(im_read_size[1] / 2)

            glands_to_mark_processed = []
            for annotation in annotations_dict_list:
                gland_name = annotation['name']
                gland_X_min = annotation['X_min']
                gland_X_max = annotation['X_max']
                gland_Y_min = annotation['Y_min']
                gland_Y_max = annotation['Y_max']

                if is_fully_contained(gland_X_min, gland_X_max, gland_Y_min, gland_Y_max,
                                       patch_X_min, patch_X_max, patch_Y_min, patch_Y_max):
                    glands_to_mark_processed.append(gland_name)

            skip_due_to_already_assigned = any([g in processed_glands for g in glands_to_mark_processed])
            if skip_due_to_already_assigned:
                pbar.update(1)
                continue

            fit_into_frame_flag = check_fit_into_frame(ref_annotation_dict=temp_center,
                                                       candidate_annotation_dict=temp_annotation_dict,
                                                       im_read_level=im_read_level,
                                                       im_read_size=im_read_size_fit_patch_level)
            if not fit_into_frame_flag:
                large_annotation_count += 1
                pbar.update(1)
                continue

            temp_img = get_img(slide=slide,
                               X_cm=temp_center['X_cm'],
                               Y_cm=temp_center['Y_cm'],
                               im_read_level=im_read_level,
                               im_read_size=im_read_size,
                               res_ratio_target_to_read=res_ratio_target_to_read)

            (temp_label_list,
             temp_mask,
             temp_binary_masks,
             temp_mask_color,
             temp_center_id,
             temp_center_checked,
             temp_error_flag,
             temp_area_list) = get_mask_complete_glands(
                annotations_dict_list=annotations_dict_list,
                center_annotation_dict=temp_center,
                im_read_level=im_read_level,
                im_read_size=im_read_size,
                res_ratio_target_to_read=res_ratio_target_to_read,
                center_gland_name=temp_annotation_dict['name'])

            if temp_error_flag:
                print('error flag', os.path.join(outdir_img, slide_id + '_' + str(num_cropped_patches) + '.png'))
                pbar.update(1)
                continue

            total_foreground_pixels = np.sum(temp_mask > 0)
            patch_total_pixels = patch_size * patch_size
            foreground_ratio = total_foreground_pixels / patch_total_pixels

            if foreground_ratio < FOREGROUND_AREA_RATIO_THRESH:
                pbar.update(1)
                continue

            if not isinstance(temp_binary_masks, np.ndarray) or temp_binary_masks.size == 0:
                pbar.update(1)
                continue

            outfile_img = os.path.join(outdir_img, slide_id + '_' + str(num_cropped_patches) + '.png')
            imageio.imwrite(outfile_img, temp_img)

            outfile_label = os.path.join(outdir_label, slide_id + '_' + str(num_cropped_patches) + '.txt')
            np.savetxt(outfile_label, np.asarray(temp_label_list, dtype=np.uint8), comments='#', delimiter='\t', fmt='%d')

            outfile_mask = os.path.join(outdir_mask, slide_id + '_' + str(num_cropped_patches) + '_mask.png')
            imageio.imwrite(outfile_mask, temp_mask)

            for j in range(temp_binary_masks.shape[0]):
                outfile_binary_mask = os.path.join(outdir_binary_masks, slide_id + '_' + str(num_cropped_patches) + '__' + str(j) + '_binary_mask.png')
                imageio.imwrite(outfile_binary_mask, temp_binary_masks[j])

            outfile_mask_color = os.path.join(outdir_mask_color, slide_id + '_' + str(num_cropped_patches) + '_mask_color.png')
            imageio.imwrite(outfile_mask_color, temp_mask_color)

            allnum = allnum + 1

            with open(cropped_patches_filelist, 'a') as f_cropped_patches_filelist:
                f_cropped_patches_filelist.write(slide_id + '\t'
                                                + str(num_cropped_patches) + '\t'
                                                + str(temp_center['X_cm'])
                                                + '\t' + str(temp_center['Y_cm'])
                                                + '\t' + str(temp_center_id)
                                                + '\t' + str(temp_center_checked)
                                                + '\t' + str(temp_label_list[temp_center_id]) + '\n')

            num_cropped_patches += 1

            for gname in glands_to_mark_processed:
                processed_glands.add(gname)

            pbar.update(1)

        pbar.close()

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not csv_exists:
                writer.writerow(csv_header)
                csv_exists = True
            writer.writerow([slide_id, num_cropped_patches])


if __name__ == "__main__":
    global allnum
    allnum = 0
    try:
        with open('tip.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    Seg(line, "tip")
                    print('已完成图片：{}的处理! 目前已经采集图片：{}'.format(line, allnum))
    except FileNotFoundError:
        print("文件未找到。")
    print("总采集图片数量：{}".format(allnum))
