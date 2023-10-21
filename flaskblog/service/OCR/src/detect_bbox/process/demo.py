# coding=utf-8
import os
from pathlib import Path
from collections import OrderedDict
import shutil
import cv2
import numpy as np
import tensorflow as tf
from flaskblog.service.OCR.src.detect_bbox.nets import model_train as model
from flaskblog.service.OCR.src.detect_bbox.utils.rpn_msr.proposal_layer import proposal_layer
from flaskblog.service.OCR.src.detect_bbox.utils.text_connector.detectors import TextDetector
from flaskblog.service.OCR.src import preprocess

MODULE_PATH   = (Path(__file__) / '../../..').resolve()
gpu = '0'
test_data_path = 'id_cards/data/demo/'
output_path = 'id_cards/data/res/'
checkpoint_path1 =  f"{MODULE_PATH}\detect_bbox\checkpoints_mlt\ctpn_50000.ckpt"
c = 0
path = 'crnn/new_demo/2/'


def preprocessing_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated_img = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    abs_diff = cv2.absdiff(gray, bg_img)
    diff_img = 255 - abs_diff
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    common_img = cv2.bilateralFilter(norm_img, 31, 11, 31)

    return common_img


def img_crop(img, box, thre, thre2):
    xmin, ymin, xmax, ymax = box
    image1 = img[int(ymin) - thre:int(ymax) + thre2, int(xmin):int(xmax)]
    return image1


def get_images3(path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def select_box(boxes, name_box):
    bbox = []
    print('len(boxes):', len(boxes))
    if len(boxes) > 1:
        Max_Area = []
        Max_Height = []
        for idx, box in enumerate(boxes):
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            Max_Area.append(Area)
            Height = max_heght(box)
            Max_Height.append(Height)
            print('Area of multi boxes', Area)
            print('Max Height of multi boxes', Height)
        if max(Max_Area) > 5500 or max(Max_Height) > 29:
            print('Max_Area', Max_Area, Max_Area.index(max(Max_Area)))
            if name_box == 'id_number':
                bbox = boxes[Max_Height.index(max(Max_Height)), :]
            else:
                bbox = boxes[Max_Area.index(max(Max_Area)), :]
        else:
            bbox = []
    else:
        try:
            box = boxes[0, :]
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            Height = max_heght(box)
            print('Area of sigle box', Area)
            print('Height of sigle box', Height)
            if name_box == 'full_name_2_fix':
                if Area > 20000:
                    bbox = boxes[0, :]
                else:
                    print('None bounding box')
            else:
                if Area > 8000 or Height > 29:
                    bbox = boxes[0, :]
                else:
                    print('None bounding box')
        except:
            bbox = boxes

    return bbox


def select_box_cmnd_new2(boxes):
    bbox = []
    print('len(boxes):', len(boxes))
    if len(boxes) > 1:
        Max_Area = []
        for idx, box in enumerate(boxes):
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = max_heght(box)
            Max_Area.append(Area)
            print('hight_selected', Area)
            # print('Max_Area',Max_Area,Max_Area.index(max(Max_Area)))
            bbox = boxes[Max_Area.index(max(Max_Area)), :]

    else:
        try:
            box = boxes[0, :]
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            Hight = max_heght(box)
            print('Area of sigle box', Area)
            print('Height of sigle box', Hight)
            if Area > 2000 and Hight > 25:
                bbox = boxes[0, :]
            else:
                print('None bounding box')
        except:
            bbox = boxes

    return bbox


def select_box_cmnd_new(boxes, name_box):
    bbox = []
    print('len(boxes):', len(boxes))
    if len(boxes) > 1:
        Max_Area = []
        for idx, box in enumerate(boxes):
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            Max_Area.append(Area)
            print('Area of multi boxes', Area)
            # print('Max_Area',Max_Area,Max_Area.index(max(Max_Area)))
            bbox = boxes[Max_Area.index(max(Max_Area)), :]
    else:
        try:
            box = boxes[0, :]
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            print('Area of sigle box', Area)
            if Area > 1000:
                bbox = boxes[0, :]
            else:
                print('None bounding box')
        except:
            bbox = boxes

    return bbox


def select_box_cccd2(boxes, name_box):
    bbox = []
    print('len(boxes):', len(boxes))
    if len(boxes) > 1:
        Max_Area = []
        for idx, box in enumerate(boxes):
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = max_heght(box)
            Max_Area.append(Area)
            print('hight_selected', Area)
        # print('Max_Area',Max_Area,Max_Area.index(max(Max_Area)))
        if (max(Max_Area)) > 10:
            bbox = boxes[Max_Area.index(max(Max_Area)), :]

    else:
        try:
            box = boxes[0, :]
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            Hight = max_heght(box)
            print('Area of sigle box', Area)
            if name_box == 'residence_2':
                print('Hight of sigle box', Hight)
                if Hight > 20:
                    bbox = boxes[0, :]
                else:
                    print('None bounding box')
            else:
                if Area > 1000:
                    bbox = boxes[0, :]
                else:
                    print('None bounding box')

        except:
            bbox = boxes

    return bbox


def select_box_cccd(boxes, name_box):
    bbox = []
    print('len(boxes):', len(boxes))
    if len(boxes) > 1:
        Max_Area = []
        for idx, box in enumerate(boxes):
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            Max_Area.append(Area)
            print('Area of multi boxes', Area)
            # print('Max_Area',Max_Area,Max_Area.index(max(Max_Area)))
            bbox = boxes[Max_Area.index(max(Max_Area)), :]
    else:
        try:
            box = boxes[0, :]
            box = box[0:8]
            box = np.expand_dims(box, axis=0)
            Area = Area_of_rectangle(box)
            print('Area of sigle box', Area)
            if Area > 1000:
                bbox = boxes[0, :]
            else:
                print('None bounding box')
        except:
            bbox = boxes

    return bbox


def Area_of_rectangle(box):
    Area = ((box[0, 5] - box[0, 1]) * (box[0, 4] - box[0, 0]))
    return Area


def max_heght(box):
    Area = (box[0, 5] - box[0, 1])
    return Area


def crop_after(image, box):
    crop_image = image[box[0, 1]:box[0, 5], box[0, 0]:box[0, 4]]
    return crop_image


def crop_after2_cmnd_new(image, name_box, box, up, left, down):
    w, h, _ = image.shape
    approximately = box[1] - up
    print(image.shape)
    print(approximately)
    print(up + abs(approximately))
    if (box[1] - up) >= 0:
        crop_image = image[box[1] - up:box[5] + down, box[0] + left:box[4] + 10]
    else:
        crop_image = image[box[1] - (up - abs(approximately)):box[5] + down, box[0]:box[4] + 10]

    return crop_image


def crop_after2_cccd(image, name_box, box, up, left, down):
    w, h, _ = image.shape
    approximately = box[1] - up
    print(image.shape)
    print(approximately)
    print(up + abs(approximately))
    if (box[1] - up) >= 0:
        crop_image = image[box[1] - up:box[5] + down, box[0] + left:box[4] + 10]
    else:
        crop_image = image[box[1] - (up - abs(approximately)):box[5] + down, box[0]:box[4] + 10]

    # crop_image = image[18:158, 86:858]
    return crop_image


def crop_after2(image, box, name_box, up, left, down):
    w, h, _ = image.shape
    approximately = box[1] - up
    print(image.shape)
    print(approximately)
    print(up + abs(approximately))
    if name_box == 'id_number':
        if (box[1] - up) >= 0:
            crop_image = image[box[1] - up:box[5] + down, :]
        else:
            crop_image = image[box[1] - (up - abs(approximately)):box[5] + down, :]
    else:
        if (box[1] - up) >= 0:
            crop_image = image[box[1] - up:box[5] + down, box[0] + left:box[4]]
        else:
            crop_image = image[box[1] - (up - abs(approximately)):box[5] + down, box[0]:box[4]]

    # crop_image = image[18:158, 86:858]
    return crop_image


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def bbox_crop(sess, img, name_box, list_fields, bbox_pred, cls_prob, input_image, input_im_info):
    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                           feed_dict={input_image: [img],
                                                      input_im_info: im_info})

    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE='H')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)

    # cost_time = (time.time() - start)
    # print("cost time: {:.2f}s".format(cost_time))
    print(name_box)
    box = select_box(boxes, name_box)
    # for i, box in enumerate(boxes):

    print('len(box)' + name_box, len(box))
    # if len(box) > 3:
    #     cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
    #                 thickness=2)
    if len(box) > 3:
        print(name_box)
        if name_box == 'full_name_1':
            img_final = crop_after2(img, box, name_box, 10, 0, 10)
            img_final = img_final[:, :, ::-1]
        elif name_box == 'full_name_1_fix':
            img_final = crop_after2(img, box, name_box, 25, 0, 25)
        elif name_box == 'full_name_2_fix':
            img_final = crop_after2(img, box, name_box, 20, 0, 20)
        elif name_box == 'date_of_birth':
            img_final = crop_after2(img, box, name_box, 10, 0, 10)
            img_final = img_final
        elif name_box == 'id_number':
            img_final = crop_after2(img, box, name_box, 5, 0, 25)
        elif name_box == 'birth_place_2':
            img_final = crop_after2(img, box, name_box, 10, 0, 10)
        elif name_box == 'residence_1' or name_box == 'birth_place_1':
            img_final = crop_after2(img, box, name_box, 10, 0, 10)
            img_final = img_final
        else:
            img_final = crop_after2(img, box, name_box, 10, 0, 10)
            img_final = img_final
        # img_final = img_final[:,:,::-1]
        list_fields[name_box] = img_final
    else:
        img_final = img

    return img_final


def main2(image, boxes):
    global rh, rw, full_name_fix, full_name_fix_2
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # os.makedirs(output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('find bounding box ...')
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):  # fix variable global
        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                model_path1 = checkpoint_path1
                saver.restore(sess, model_path1)
                list_fields = OrderedDict()
                boxes['full_name_1_fix'] = []
                boxes['full_name_2_fix'] = []
                for name_box, box in boxes.items():
                    if name_box == 'id_number':
                        img = img_crop(image, box, 15, 20)
                    elif name_box == 'day_of_birth':
                        img = img_crop(image, box, 10, 10)
                        img, (rh, rw) = resize_image(img)
                    elif name_box == 'full_name_2':
                        img = img_crop(image, box, 0, 0)
                        full_name_fix_2 = img.copy()
                        img = img[:, :, ::-1]
                    elif name_box == 'full_name_2_fix':
                        full_name_fix_2, (rh, rw) = resize_image(full_name_fix_2)
                        img = full_name_fix_2[:, :, ::-1]
                    elif name_box == 'full_name_1':
                        img = img_crop(image, box, 15, 15)
                        full_name_fix = img.copy()
                        img = img[:, :, ::-1]
                    elif name_box == 'full_name_1_fix':
                        full_name_fix, (rh, rw) = resize_image(full_name_fix)
                        img = full_name_fix[:, :, ::-1]
                    elif name_box == 'residence_2' or name_box == 'birth_place_2':
                        img = img_crop(image, box, 15, 20)
                        img = img[:, :, ::-1]
                        h1, w1, _ = img.shape
                    else:
                        img = img_crop(image, box, 10, 10)
                        img = img[:, :, ::-1]
                    cv2.imwrite("OCR/src/detect_bbox/process/debug/" + name_box + ".jpg", img)
                    img_final = bbox_crop(sess, img, name_box, list_fields, bbox_pred, cls_prob, input_image,
                                          input_im_info)
                    cv2.imwrite("OCR/src/detect_bbox/process/debug/" + name_box + "_after.jpg", img_final)
            return list_fields


def bbox_crop_cmnd_new(sess, img, birth_place_2_fix, name_box, list_fields, bbox_pred, cls_prob, input_image,
                       input_im_info):
    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                           feed_dict={input_image: [img],
                                                      input_im_info: im_info})

    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE='H')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)

    # cost_time = (time.time() - start)
    # print("cost time: {:.2f}s".format(cost_time))
    print(name_box)
    if name_box == 'birth_place_2' or name_box == 'residence_2' or name_box == 'birth_place_1':
        box = select_box_cmnd_new2(boxes)
    else:
        box = select_box_cmnd_new(boxes, boxes)
    # for i, box in enumerate(boxes):

    print('len(box)' + name_box, len(box))
    # if len(box) > 3:
    #     cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
    #                 thickness=2)

    if len(box) > 3:
        print(name_box)
        if name_box == 'id_number':
            img_final = crop_after2_cmnd_new(img, name_box, box, 10, 0, 30)
            img_final = img_final[:, :, ::-1]
            list_fields[name_box] = img_final
        elif name_box == 'gender':
            img_final = crop_after2_cmnd_new(img, name_box, box, 7, 0, 5)
            img_final = img_final[:, :, ::-1]
            list_fields[name_box] = preprocessing_bbox(img_final)
        elif name_box == 'ethnicity':
            img_final = crop_after2_cmnd_new(img, name_box, box, 5, 0, 5)
            img_final = img_final[:, :, ::-1]
            list_fields[name_box] = preprocessing_bbox(img_final)
        elif name_box == 'full_name_1':
            img_final = crop_after2_cmnd_new(img, name_box, box, 10, 0, 10)
            list_fields[name_box] = img_final
        elif name_box == 'birth_place_2':
            # cv2.imshow('birth_place_2_fix',birth_place_2_fix)
            img_final = crop_after2_cmnd_new(birth_place_2_fix, name_box, box, 10, 0, 22)
            list_fields[name_box] = img_final
        elif name_box == 'date_of_birth':
            img_final = crop_after2_cmnd_new(img, name_box, box, 10, 0, 10)
            img_final = img_final[:, :, ::-1]
            list_fields[name_box] = img_final[:, :, ::-1]
        elif name_box == 'residence_1':
            img_final = crop_after2_cmnd_new(img, name_box, box, 10, 0, 10)
            list_fields[name_box] = preprocessing_bbox(img_final)
            list_fields[name_box] = img_final
        elif name_box == 'residence_2':
            img_final = crop_after2_cmnd_new(img, name_box, box, 15, 0, 10)
            img_final = img_final
            list_fields[name_box] = img_final
        else:
            img_final = crop_after2_cmnd_new(img, name_box, box, 10, 0, 10)
            list_fields[name_box] = img_final
    else:
        img_final = img
    return img_final


def main_cmnd_new(image, boxes):
    global rh, rw
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # os.makedirs(output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('find bounding box ...')
    birth_place_2_fix = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):  # fix variable global
        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                model_path1 = checkpoint_path1
                saver.restore(sess, model_path1)
                list_fields = OrderedDict()
                for name_box, box in boxes.items():
                    if name_box == 'birth_place_2':
                        img = img_crop(image, box, 5, 5)
                        img = img[:, :, ::-1]
                        image1 = image.copy()
                        birth_place_2_fix = img_crop(image1, box, 20, 15)
                    elif name_box == 'id_number':
                        img = img_crop(image, box, 10, 20)
                        img = img[:, :, ::-1]
                    else:
                        img = img_crop(image, box, 5, 7)
                        img = img[:, :, ::-1]
                    cv2.imwrite("OCR/src/detect_bbox/process/debug/" + name_box + ".jpg", img)
                    img_final = bbox_crop_cmnd_new(sess, img, birth_place_2_fix, name_box, list_fields, bbox_pred,
                                                   cls_prob, input_image, input_im_info)
                    cv2.imwrite("OCR/src/detect_bbox/process/debug/" + name_box + "_after.jpg", img_final)

            return list_fields


def bbox_crop_cccd(sess, img, birth_place_2_fix, name_box, list_fields, bbox_pred, cls_prob, input_image,
                   input_im_info):
    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                           feed_dict={input_image: [img],
                                                      input_im_info: im_info})
    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE='H')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)

    # cost_time = (time.time() - start)
    # print("cost time: {:.2f}s".format(cost_time))
    print(name_box)
    if name_box == 'birth_place_2' or name_box == 'residence_2' or name_box == 'birth_place_1' or name_box == 'gender':
        box = select_box_cccd2(boxes, name_box)
    else:
        box = select_box_cccd(boxes, boxes)

    print('len(box)' + name_box, len(box))

    if len(box) > 4:
        print(name_box)
        if name_box == 'id_number':
            img_final = crop_after2_cccd(img, name_box, box, 10, 0, 40)
            list_fields[name_box] = img_final
        elif name_box == 'gender':
            img_final = crop_after2_cccd(img, name_box, box, 5, 0, 5)
            # img_final = img_final[:, :, ::-1]
            list_fields[name_box] = preprocessing_bbox(img_final)
        elif name_box == 'full_name_1':
            img_final = crop_after2_cccd(img, name_box, box, 8, 0, 8)
            list_fields[name_box] = img_final
        elif name_box == 'date_of_birth':
            img_final = crop_after2_cccd(img, name_box, box, 10, 0, 10)
            list_fields[name_box] = img_final
        elif name_box == 'birth_place_2':
            img_final = crop_after2_cccd(birth_place_2_fix, name_box, box, 15, -10, 25)
            list_fields[name_box] = preprocessing_bbox(img_final)
        elif name_box == 'residence_1':
            img_final = crop_after2_cccd(img, name_box, box, 8, 0, 5)
            list_fields[name_box] = img_final
        else:
            img_final = crop_after2_cccd(img, name_box, box, 10, 0, 8)
            list_fields[name_box] = preprocessing_bbox(img_final)
    else:
        img_final = img
    return img_final


def main_cccd(image, boxes):
    global rh, rw
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('find bounding box ...')
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):  # fix variable global
        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
            birth_place_2_fix = []
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                model_path1 = checkpoint_path1
                saver.restore(sess, model_path1)
                list_fields = OrderedDict()

                for name_box, box in boxes.items():
                    if name_box == 'id_number':
                        img = img_crop(image, box, 10, 33)
                    elif name_box == 'birth_place_2':
                        img = img_crop(image, box, 5, 7)
                        image1 = image.copy()
                        birth_place_2_fix = img_crop(image1, box, 22, 15)
                    elif name_box == 'residence_1':
                        img = img_crop(image, box, 10, 10)
                        # img = img[:, :, ::-1]
                    elif name_box == 'residence_2':
                        img = img_crop(image, box, 10, 10)
                        # img = img[:, :, ::-1]
                    else:
                        img = img_crop(image, box, 10, 8)

                    cv2.imwrite("OCR/src/detect_bbox/process/debug/" + name_box + ".jpg", img)
                    img_final = bbox_crop_cccd(sess, img, birth_place_2_fix, name_box, list_fields, bbox_pred, cls_prob,
                                               input_image, input_im_info)
                    cv2.imwrite("OCR/src/detect_bbox/process/debug/" + name_box + "_after.jpg", img_final)

            return list_fields


if __name__ == '__main__':
    image = cv2.imread(
        '/home/ncongthanh/Desktop/work_everyday/new/py-web-ocr_29_06_2020/py-web-ocr_full/py-web-ocr/OCR_TIC/data_testing/ten333.png')
    img = main2(image)
