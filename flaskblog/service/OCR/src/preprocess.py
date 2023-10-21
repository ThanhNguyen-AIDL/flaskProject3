from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image

from flaskblog.service.OCR.src.config import config
from flaskblog.service.OCR.src.utils import img_from_bbox, find_text_bbox


################################################################################
################################################################################
##    ###  ##### #   # ####  #     #####       #     #####  #### #   # #####  ##
##   #       #   ## ## #   # #     #           #       #   #     #   #   #    ##
##    ###    #   # # # ####  #     #####       #       #   #  ## #####   #    ##
##       #   #   #   # #     #     #           #       #   #   # #   #   #    ##
##    ###  ##### #   # #     ##### #####       ##### #####  #### #   #   #    ##
################################################################################
################################################################################


def enhance_edge(img,times=1):
    #generating the kernels
    kernel = np.array([
        [-1,-1,-1,-1,-1],
        [-1, 2, 2, 2,-1],
        [-1, 2, 9, 2,-1],
        [-1, 2, 2, 2,-1],
        [-1,-1,-1,-1,-1],
        ]) / 9.0
    img1 = img.copy()
    for _ in range(times):
        img1 = cv2.filter2D(img1, -1, kernel)
    return img


def change_gamma(img, gamma = 1.2):
    # change image gamma. Not being used.
    
    raise DeprecationWarning

    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    img = cv2.LUT(img, lookUpTable)
    return img

    
def adjust_gamma(image, gamma=1.0):
    #build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def calculate_brightness(img_np):
    # image = Image.open(image_path)
    img = Image.fromarray(img_np)
    greyscale_image = img.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

def preprocess(img_np):
    # img_brg = cv2.imread(img_path)
    # a = calculate_brightness(img_np)
    img = adjust_gamma(img_np)
    return img
    
    
def equalize_light(img, limit=3, grid=(7,7), gray=False):

    if img.ndim == 2:
        img  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe   = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl      = clahe.apply(l)
    limg    = cv2.merge((cl,a,b))
    final   = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


################################################################################
################################################################################
##    #####  ####   #####  #   #         #####  #   #    #     ####  #####    ##
##      #    #   #    #    ## ##           #    ## ##   # #   #      #        ##
##      #    ####     #    # # #           #    # # #  #####  #  ##  #####    ##
##      #    # #      #    #   #           #    #   #  #   #  #   #  #        ##
##      #    #  ##  #####  #   #         #####  #   #  #   #   ####  #####    ##
################################################################################
################################################################################


def crop_horizontally(img, rel_padding=0.04, peak_height=0.3, 
                      peak_prominence=0.5, debug=False):
    
    h, w = img.shape[:2]
    padding = int(rel_padding * w)
    
    bboxes = find_text_bbox(img,
        axis_to_crop     = 1,
        deriv_order      = 0,
        focus_vertical   = (0.25,0.75),
        peak_height      = peak_height,
        peak_prominence  = peak_prominence,
        padding          = padding,
        debug            = debug)
        
    if len(bboxes) == 0:
        return 0, w

    start = bboxes[0][0]              # xmin of first bbox
    end   = bboxes[-1][0] + 2*padding # xmin of last bbox with 2 padding
           
    return start, end


def crop_id_number(img, debug=False):
    start, end = crop_horizontally(img, peak_height=0.3, peak_prominence=0.5, debug=debug)
    trimmed_img = img[:,start:end]
    return trimmed_img


def crop_date_of_birth(img, debug=False):
    start, end = crop_horizontally(img, peak_height=1, peak_prominence=0.7, debug=debug)
    trimmed_img = img[:,start:end]
    return trimmed_img


################################################################################
################################################################################
##                 ####   #   #    ###    #####    ###    #   #               ##
##                #       #   #   #         #     #   #   ## ##               ##
##                #       #   #    ###      #     #   #   # # #               ##
##                #       #   #       #     #     #   #   #   #               ##
##                 ####    ###     ###      #      ###    #   #               ##
################################################################################
################################################################################


def process_1_front(img, mapped_box_dict, debug=False):
    cv2.imshow('process_1_front',img)
    cv2.waitKey()
    # GENERAL PROCESSING
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated_img = cv2.dilate(gray, np.ones((5,5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)

    abs_diff = cv2.absdiff(gray, bg_img)
    diff_img = 255 - abs_diff
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, 
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    common_img = cv2.bilateralFilter(norm_img,31,11,31)
    
    bbox_configs = config.get_bbox_configs('1', 'front')
    all_blocks = bbox_configs.keys()
    block_img_dict = OrderedDict()
    for block_name in all_blocks:
        # all blocks will share these bbox and images, but some will 
        # have customizations
        bbox = mapped_box_dict.get(block_name,None)
        if bbox is None:
            block_img_dict[block_name] = None
            continue
        
        if block_name == 'id_number':
            block_img = img_from_bbox(img,bbox)
            block_img = crop_id_number(block_img)
            block_img_dict[block_name] = block_img
            cv2.imshow(block_name,block_img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        elif block_name == 'date_of_birth':
            block_img = img_from_bbox(img,bbox)
            block_img = crop_date_of_birth(block_img)
            block_img_dict[block_name] = block_img
            cv2.imshow(block_name, block_img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        else:
            block_img     = img_from_bbox(common_img,bbox)

            raw_block_img = img_from_bbox(img,bbox)
            start,end     = crop_horizontally(raw_block_img, peak_height=1, peak_prominence=0.7,debug=debug)
            block_img     = block_img[:,:]
            block_img_dict[block_name] = common_img

            cv2.imshow(block_name,block_img)
            cv2.waitKey()


    return dict(block_img_dict)

def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated_img = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)

    abs_diff = cv2.absdiff(gray, bg_img)
    diff_img = 255 - abs_diff
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    common_img = cv2.bilateralFilter(norm_img, 31, 11, 31)
    return common_img


def process_1_front_2(img, mapped_box_dict, debug=False):
    # ['id_number','full_name_1', 'full_name_2', 'date_of_birth', 'birth_place_1',
    # 'birth_place_2', 'residence_1', 'residence_2']
    cv2.imshow('process_1_front', img)
    cv2.waitKey()
    # GENERAL PROCESSING
    block_img_dict = OrderedDict()
    for block_name,  block_img in mapped_box_dict.items():
        if block_name == 'id_number':
            block_img = crop_id_number(block_img)
            block_img_dict[block_name] = block_img
            cv2.imshow(block_name, block_img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        elif block_name == 'date_of_birth':
            block_img = crop_date_of_birth(block_img)
            block_img_dict[block_name] = block_img
            cv2.imshow(block_name, block_img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        else:
            block_img_dict[block_name] = block_img

        cv2.imshow(block_name, block_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return dict(block_img_dict)
def process_1_back(img, mapped_box_dict, debug=False):
    return block_img_dict
def process_2_front(img, mapped_box_dict, debug=False):
    return block_img_dict
def process_2_back(img, mapped_box_dict, debug=False):
    return block_img_dict

def process_3_front(img, mapped_box_dict, debug=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 
                                maxValue=255, 
                                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
                                thresholdType=cv2.THRESH_BINARY,
                                blockSize=15,
                                C=15)
    
    
    # query_image = cv2.cvtColor(cv2.bitwise_not(query_image), cv2.COLOR_BGR2GRAY)
    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(query_image, None, None, None, 4,
    #                                                                      cv2.CV_32S)
    # sizes = stats[1:, -1]  # get CC_STAT_AREA component
    # img2 = np.zeros((labels.shape), np.uint8)

    # for i in range(0, nlabels - 1):
    #     if sizes[i] >= 20:  # filter small dotted regions
    #         img2[labels == i + 1] = 255
    # query_image = cv2.bitwise_not(img2)
    
    # from dhp import imshow
    # imshow(img)
    # imshow(cv2.bitwise_not(img))
    img = cv2.bitwise_not(img)
    nlabels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(img, None, None, None, 4, cv2.CV_32S)
    sizes = stats[1:, -1]  # get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 20:  # filter small dotted regions
            img2[labels == i + 1] = 255
    common_img = cv2.bitwise_not(img2)
    
    bbox_configs = config.get_bbox_configs('3', 'front')
    all_blocks = bbox_configs.keys()
    block_img_dict = OrderedDict()
    for block_name in all_blocks:
        bbox = mapped_box_dict.get(block_name,None)
        if bbox is None:
            block_img_dict[block_name] = None
            continue
        
        block_img = img_from_bbox(common_img,bbox)
        block_img_dict[block_name] = block_img
    return block_img_dict

def process_3_back(img, mapped_box_dict, debug=False):
    return block_img_dict


def _get_img_preprocessor(card_type,side):
    processors = {
        '1': {
            'front' : process_1_front_2,
            'back'  : process_1_back,
        },
        '2': {
            'front' : process_2_front,
            'back'  : process_2_back,
        },
        '3': {
            'front' : process_3_front,
            'back'  : process_3_back,
        },
    }
    processor = processors[card_type][side]
    return processor


def preprocess_img(img,mapped_box_dict,card_type,side,debug=False):
    """interface"""
    processor      = _get_img_preprocessor(card_type,side)
    block_img_dict = processor(img, mapped_box_dict,debug=debug)
    return block_img_dict

if __name__ == '__main__':
    path = 'OCR/data_customer/test/0-0 (1).jpg'
    image = cv2.imread(path)
    image1 = preprocess(image)
    cv2.imshow('image_orginal',image)
    cv2.imshow('image_after_preprocessing',image1)
    cv2.waitKey()
    cv2.destroyAllWindows()