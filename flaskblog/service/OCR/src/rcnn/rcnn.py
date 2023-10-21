# import string
# import argparse
from argparse import Namespace

from PIL import Image
import torch
import torch.utils.data
import cv2

from flaskblog.service.OCR.src.config import config
from flaskblog.service.OCR.src.rcnn.dataset import AlignCollate    # RawDataset
from flaskblog.service.OCR.src.rcnn.model import Model
from flaskblog.service.OCR.src.rcnn.utils import CTCLabelConverter # AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RCNN_OCR:
    def __init__(self, rcnn_config):
        "init parameter "

        if rcnn_config=='number_model':
            weight = config.RCNN_NUM_WEIGHT
            whitelist  = u'-/0123456789'
        
            opt = Namespace(batch_max_length = 25,imgH = 32,imgW = 100,
                    rgb = True,character='-/0123456789',Transformation='None',
                    keep_ratio_with_pad = 'None',num_class = 13,
                    FeatureExtraction = "ResNet", SequenceModeling = 'None',
                    Prediction = "CTC",input_channel = 3,output_channel= 512,
                    hidden_size = 256,saved_model = weight,
                    workers=4, sensitive=False,num_fiducial=20,batch_size=1)
            
        elif rcnn_config=='mixed_model':
            weight = config.RCNN_MIX_WEIGHT
            whitelist  = u'-/0123456789abcdefghijklmnopqrstuvwxyzabcdefghijklmno'
            whitelist += u'pqrstuvwxyzàáâãèéêìíðòóôõöùúüýàáâãèéêìíðòóôõöùúüýāāăă'
            whitelist += u'đđĩĩōōũũūūơơưưạạảảấấầầẩẩẫẫậậắắằằẳẳẵẵặặẹẹẻẻẽẽếếềềểểễễệ'
            whitelist += u'ệỉỉịịọọỏỏốốồồổổỗỗộộớớờờởởỡỡợợụụủủứứừừửửữữựựỳỳỵỵỷỷỹỹ'
            
            opt = Namespace(
                batch_max_length = 25, imgH = 32, imgW = 100, rgb = True, 
                character= whitelist, Transformation='TPS', num_class = 211, 
                FeatureExtraction = "ResNet", SequenceModeling = 'BiLSTM', 
                Prediction = "CTC", input_channel = 3, output_channel= 512, 
                hidden_size = 256, saved_model = weight, workers=4, 
                num_fiducial=20, batch_size=1)

        self.converter = CTCLabelConverter(whitelist)
        self.model = Model(opt)
        self.model = torch.nn.DataParallel(self.model).to(device)
        self.model.load_state_dict(torch.load(weight, map_location=device))
        self.AlignCollate_demo = AlignCollate(imgH=32, imgW=100)

        self.batch_max_length = 25
    
    def predict(self,image):
        image_tensor = self.AlignCollate_demo(image)
        # image.show()
        self.model.eval()
        with torch.no_grad():
            batch_size = image_tensor.size(0)
            image_tensor = image_tensor.to(device)
            length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(device)
            preds = self.model(image_tensor, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)[0]
        print('result_RCNN_number',preds_str)
        return preds_str


if __name__ == '__main__':
    from time import time as t
    # t1 = t()
    # test = DemoOCR()
    # t2 = t()
    # # image = cv2.imread("demo_image/2.png")
    # from dhp import walk_path
    # paths = walk_path(f"src/crnn/demo_image/")
    # for pathImage in paths:
    #     print(pathImage)
    #     # pathImage = f"demo_image/{i}{i}.png"
    #     image = Image.open(pathImage).convert('RGB')
    #     t3 = t()
    #     result = test.predict(image)
    #     t4 = t()
    #     print(result)
    # print(t2-t1, t4-t3)


    pathImage = '/home/ncongthanh/Desktop/work_everyday/new/py-web-ocr_29_06_2020/py-web-ocr_full/py-web-ocr/OCR_TIC/samples/test/ten.png'
    image = cv2.imread(pathImage)
    img = Image.fromarray(image)
    img = img.convert('RGB')
    t3 = t()
    info = 'number_model'
    model = RCNN_OCR(info)
    result = model.predict(img)
    # result = RCNN_flaskblog.service.OCR.predict(image)
    t4 = t()
    print(result)


