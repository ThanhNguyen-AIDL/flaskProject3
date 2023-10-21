import os
import subprocess
import sys
import re
from collections import OrderedDict

import pytesseract
from PIL import Image
from flaskblog.service.OCR.src import preprocess
from flaskblog.service.OCR.src.config import config
from flaskblog.service.OCR.src.utils import SingletonMeta
from flaskblog.service.OCR.src.rcnn.rcnn import RCNN_OCR
import numpy as np
import cv2
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
global flat1, flat2


def clean_strings(texts, has_char=True, has_num=True,
                  allowed_symbols='/\-.,', split_CamelCase=True,
                  support_vnese=True):
    """Clean up OCR string result
    Example:
    >>> test = '  $%$  123 - ^&* NgọcHà,{P}húMỹ biênHòa hà nội|  %$^%$  '
    >>> clean_strings(test)
    '123 - Ngọc Hà, Phú Mỹ biên Hòa hà nội'
    """

    vi_up = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
    vi_lo = u'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'

    up_pattern = vi_up + 'A-Z' if support_vnese else 'A-Z'
    low_pattern = vi_lo + 'a-z' if support_vnese else 'a-z'
    all_chars = up_pattern + low_pattern
    if allowed_symbols is True:
        allowed_symbols = '\W'

    # first clean up
    if has_char and has_num:
        texts = re.sub(f'[^{all_chars}0-9{allowed_symbols} ]', '', texts)
    elif has_char and not has_num:  # a.k.a text only
        texts = re.sub(f'[^{all_chars}{allowed_symbols} ]', '', texts)
    elif not has_char and has_num:  # a.k.a number only
        texts = re.sub(f'[^0-9{allowed_symbols} ]', '', texts)
    else:
        raise ValueError('Either has_num or has_char must be True')

    if split_CamelCase:
        # make CamelCase separated
        pattern = f"[{up_pattern}][{low_pattern}]+|\d+|[{allowed_symbols}]+|[{low_pattern}]+|\w+"
        ret = re.findall(r"{}".format(pattern), texts)
        ret = list(filter(lambda x: x != ' ', ret))
        texts = ' '.join(ret)

    # formating
    texts = texts.replace('  ', ' ')
    texts = texts.replace(' ,', ',')
    # below line means remove symbols at the end or at the start of line
    texts = re.sub(f"^[{allowed_symbols}]+|[{allowed_symbols}]+$", '', texts)
    texts = texts.strip()
    return texts


def clean_number_only(texts):
    # has_num=True because it's better for text correction later
    return clean_strings(texts,
                         allowed_symbols='',  # `|\{\}\[\]\
                         split_CamelCase=False)


def clean_text_only(texts):
    return clean_strings(texts,
                         allowed_symbols=' ',
                         has_num=False)


def clean_date(texts):
    return clean_strings(texts,
                         allowed_symbols='/\-',
                         split_CamelCase=False)


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


def clean_mix(texts):
    pattern = r"\.\.+"
    result_string = re.sub(pattern, '', texts)
    result_string = clean_strings(result_string, allowed_symbols='/\-,.')
    return result_string


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1


def correct_birth(number):
    number_correct = []
    number_correct[:0] = number
    if number_correct[2] != '/':
        if number_correct[2] == '-':
            number_correct[2] = '/'
        else:
            number_correct.insert(2, '/')
    if number_correct[-5] != '/':
        if number_correct[-5] == '-':
            number_correct[-5] = '/'
        else:
            number_correct.insert(-4, '/'),

    number_correct = listToString(number_correct)

    print('number_correct:{}'.format(number_correct))

    return number_correct


def Nguyen(text):
    correct_Nguyen = []
    name = []
    for idx, i in enumerate(text):
        if idx < 6:
            correct_Nguyen.append(i)
        else:
            name.append(i)
    name = listToString(name)
    correct_Nguyen = listToString(correct_Nguyen)
    if correct_Nguyen == 'NGUYÊN' or correct_Nguyen == 'NGUYEN':
        result = 'NGUYỄN' + name
    else:
        result = correct_Nguyen + name
    return result


def Tran(text):
    correct_Tran = []
    name = []
    for idx, i in enumerate(text):
        if idx < 4:
            correct_Tran.append(i)
        else:
            name.append(i)
    name = listToString(name)
    if listToString(correct_Tran[:2]) == 'TR' and listToString(correct_Tran[3]) == 'N':
        result = 'TRẦN' + name
    else:
        correct_Tran = listToString(correct_Tran)
        result = correct_Tran + name
    return result


def correct_name(text):
    result = Nguyen(text)
    result = Tran(result)

    return result


def output_image(image, index, name_bouding_box):
    parth_dir = 'crop_bounding_box'
    cv2.imwrite(str(parth_dir / (name_bouding_box + '_' + '.png')), image)


class OCR(metaclass=SingletonMeta):
    def __init__(self):
        self.rcnn_number_ocr = RCNN_OCR(rcnn_config='number_model')
        self.rcnn_mixed_ocr = RCNN_OCR(rcnn_config='mixed_model')

    @staticmethod
    def _get_config_and_clean_function(block_name):
        block_types = {  # key: value = block_name: block_type
            'id_number': 'number_only',

            'date_of_birth': 'date',
            'issue_day': 'date',

            'full_name_1': 'text_only',
            'full_name_2': 'text_only',
            'full_name_1_fix': 'text_only',
            'full_name_2_fix': 'text_only',
            'other_name': 'text_only',
            'father_name': 'text_only',
            'mother_name': 'text_only',
            'nation': 'text_only',
            'ethnicity': 'text_only',
            'religion': 'text_only',
            'gender': 'text_only',

            'birth_place_1': 'mixed',
            'birth_place_2': 'mixed',
            'residence_1': 'mixed',
            'residence_2': 'mixed',
            'features_1': 'mixed',
            'features_2': 'mixed',
            'issue_loc': 'mixed',
        }
        # TODO: move to config
        type_configs = {  # key: value = block_type : tess_config
            # 'number_only' : f'-l eng --oem 1 --psm 10', # 0.750318
            'number_only': ('rcnn', 'number_model'),  # 0.984127
            'date': ('rcnn', 'number_model'),  # 0.950231
            # 'text_only'   : ('rcnn', 'mix_model'),
            # 'mixed'       : ('rcnn', 'mix_model'),

            'text_only': ('tess', '-l vie --oem 1 --psm 7'),
            'mixed': ('tess', '-l vie --oem 1 --psm 10'),
        }

        first_clean_func = {  # key: value = block_type : clean_function
            'number_only': clean_number_only,
            'date': clean_date,
            'text_only': clean_text_only,
            'mixed': clean_mix,
        }

        block_type = block_types[block_name]
        ocr_config = type_configs[block_type]
        clean_function = first_clean_func[block_type]
        return ocr_config, clean_function

    @staticmethod
    def run_tesseract_ocr(img, block_name, tess_config, tess_path=None, convert_to_std_result=False):

        if tess_path is None:
            tess_path = config.TESSERACT_PATH
        if not os.path.exists(tess_path):
            if sys.platform == 'linux':
                cli = 'which tesseract'
                tess_path = subprocess.check_output(cli, shell=True).decode()[:-1]
            elif sys.platform == 'win32':
                pass
            elif sys.platform == 'darwin':
                pass

        pytesseract.pytesseract.tesseract_cmd = tess_path
        # TODO: 2 - split words by distance

        # img = preprocessing(img)
        # cv2.imshow('result_pytesseract',img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        result = pytesseract.image_to_string(img, config=tess_config)
        print('result_of_' + block_name, result)
        # ouput_result_of_image
        # output_image(img,block_name)

        if block_name == 'full_name_1':
            result = correct_name(result)
            print('result_name_corrected: {}'.format(result))
        return result

    def run_rcnn_ocr(self, block_img, block_name, rcnn_config):
        if block_name == 'id_number':
            begin, end = preprocess.crop_horizontally(block_img)
            block_img = block_img[:, begin:end]
            # cv2.imshow('',block_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        img = Image.fromarray(block_img)
        # img = img.convert('RGB')

        if rcnn_config == 'number_model':

            result = self.rcnn_number_ocr.predict(img)
            if block_name == 'date_of_birth':
                result = correct_birth(result)


        elif rcnn_config == 'mix_model':
            result = self.rcnn_mixed_ocr.predict(img)
        else:
            raise ValueError('incorrect RCNN configuration')

        return result

    @staticmethod
    def merge_blocks(parsed_dict, card_type, side, split_by=', '):
        """groups blocks like "full_name_1", "full_name_2" into "full_name" with
        combined values
        """
        result = OrderedDict()
        merged_keys = []

        default_keys = config.get_bbox_configs(card_type, side).keys()
        if card_type == '1':
            default_keys = list(default_keys)[:8]
        print('default_keys', default_keys)
        for block_name in default_keys:

            value = parsed_dict.get(block_name, '')
            trimmed_name = block_name[:-2]

            if not block_name[-1:].isdigit():
                result[block_name] = value
            elif trimmed_name not in merged_keys:
                rel_keys = sorted(list(filter(lambda x: trimmed_name in x,
                                              parsed_dict)))
                rel_vals = [parsed_dict[k] for k in rel_keys if parsed_dict[k] != '']
                merged_val = split_by.join(rel_vals)
                result[trimmed_name] = merged_val
                merged_keys.append(trimmed_name)
            else:
                continue
        return dict(result)

    def parse_text(self, block_img_dict, card_type, side, debug=False,
                   custom_config: tuple = None):
        """interface"""
        flat1 = True
        flat2 = True
        parsed_dict = {}
        threshold_text = 6
        for block_name, block_img in block_img_dict.items():
            if block_img is None:
                continue
            ocr_config, clean_function = self._get_config_and_clean_function(block_name)
            if custom_config is not None:
                ocr_config = custom_config
            # select model and pass config
            model, model_config = ocr_config
            if model == 'rcnn':
                ocr_result = self.run_rcnn_ocr(block_img, block_name, rcnn_config=model_config)
            elif model == 'tess':
                ocr_result = self.run_tesseract_ocr(block_img, block_name, tess_config=model_config)
            if card_type == '1':
                ocr_result = clean_function(ocr_result)
                if block_name == 'full_name_1':
                    correct_result = self.run_tesseract_ocr(block_img, block_name, tess_config=model_config)
                    if len(ocr_result) > threshold_text:
                        parsed_dict[block_name] = ocr_result
                        flat1 = False
                    elif len(correct_result) > threshold_text:
                        parsed_dict[block_name] = correct_result
                        flat1 = False
                    else:
                        flat1 = True
                elif block_name == 'full_name_2':
                    correct_result2 = self.run_tesseract_ocr(block_img, block_name, tess_config=model_config)
                    if len(ocr_result) > threshold_text:
                        parsed_dict[block_name] = ocr_result
                        flat2 = False
                    elif len(correct_result2) > threshold_text:
                        parsed_dict[block_name] = correct_result2
                        flat2 = False
                    else:
                        flat2 = True
                elif block_name == 'full_name_1_fix' and flat1:
                    ocr_result_fix = self.run_tesseract_ocr(preprocessing(block_img), 'ocr_result_fix',
                                                            tess_config=model_config)
                    if len(ocr_result) > threshold_text:
                        parsed_dict['full_name_1'] = ocr_result
                    if len(ocr_result_fix) > threshold_text:
                        parsed_dict['full_name_1'] = ocr_result_fix
                elif block_name == 'full_name_2_fix' and flat2:
                    ocr_result_fix2 = self.run_tesseract_ocr(preprocessing(block_img), 'ocr_result_fix2',
                                                             tess_config=model_config)
                    if len(ocr_result) > threshold_text:
                        parsed_dict['full_name_2'] = ocr_result
                    if len(ocr_result_fix2) > threshold_text:
                        parsed_dict['full_name_2'] = ocr_result_fix2
                else:
                    parsed_dict[block_name] = ocr_result
            if card_type == '2':
                if block_name == 'full_name_1':
                    ocr_result = clean_function(ocr_result)
                    parsed_dict[block_name] = ocr_result
                else:
                    ocr_result = clean_function(ocr_result)
                    parsed_dict[block_name] = ocr_result
            if card_type == '3':
                if block_name == 'full_name_1':
                    ocr_result = clean_function(ocr_result)
                    parsed_dict[block_name] = ocr_result
                else:
                    ocr_result = clean_function(ocr_result)
                    parsed_dict[block_name] = ocr_result
        if not flat1:
            parsed_dict.pop('full_name_1_fix')
        if not flat2:
            parsed_dict.pop('full_name_2_fix')
        print('parsed_dict_before:', parsed_dict)
        parsed_dict = self.merge_blocks(parsed_dict, card_type, side, )
        print('parsed_dict_after: ', parsed_dict)
        return parsed_dict
