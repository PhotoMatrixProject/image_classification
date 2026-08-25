import cv2
from skimage import feature
import numpy as np
import torchvision
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch
from collections import Counter
from PIL import Image
import os
import sys
import shutil

radius = 3
n_points = radius * 8

# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
#     return os.path.join(base_path, relative_path)

class ClassificationTool:
    def __init__(self, classification_path):
        self.classification_path = classification_path
        self.class_names = ['ad', 'not_ad']
        # TODO: check if model exists, if not - download it from the server???
        self.big_model = self._load_model_vgg16(r'.\model\torch_model_vgg16_step4_lbp_more_data.pt', 2)
        
    def _load_model_vgg16(self, weight_path:str, classes_num:int):
        model = models.vgg16_bn()   
        model.classifier[-1] = nn.Linear(4096, classes_num, bias=True)

        pretrained_weights = model.features[0].weight
        new_featres = nn.Sequential(*list(model.features.children()))
        new_featres[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        new_featres[0].weight.data.normal_(0, 0.001)
        new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)


        model.features = new_featres
        model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=torch.device('cpu') ))
        model.eval()
        return model
    
    def _load_model_vgg19(self, weight_path:str, classes_num:int):
        model = models.vgg19_bn()   
        model.classifier[-1] = nn.Linear(4096, classes_num, bias=True)

        pretrained_weights = model.features[0].weight
        new_featres = nn.Sequential(*list(model.features.children()))
        new_featres[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        new_featres[0].weight.data.normal_(0, 0.001)
        new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)


        model.features = new_featres
        model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=torch.device('cpu') ))
        model.eval()
        # print(model)
        return model
    
    def _stretch_contrast(self, img_path:str):
        '''Stretches contrast of the given image, returns modified image.'''
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # print(image.shape)
            ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
            y_channel_stretched = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
            contrast_stretched_ycrcb = cv2.merge([y_channel_stretched, cr_channel, cb_channel])
            contrast_stretched_image = cv2.cvtColor(contrast_stretched_ycrcb, cv2.COLOR_YCrCb2BGR)
            return contrast_stretched_image
        except:
            print('IMG ERROR', img_path)
    
    def _extract_lbp(self, img_path:str):
        try:
            '''Extracts local binary pattern features from the given image, return LBP features.'''
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # for lbp
            lbp = feature.local_binary_pattern(gray_img, n_points, radius, method='uniform')
            lbp = lbp.astype(np.uint8)

            return lbp
        except:
            print('IMG ERR', img_path)
    
    def _get_fft(self, img_path:str):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return img_back.astype(np.float32)
    
    def _transform_image(self, img_path:str):
        try:
            contrast_stretched_image = self._stretch_contrast(img_path)
            # fft = self._get_fft(img_path)
            lbp = self._extract_lbp(img_path)
            # edges = self._extract_edges(img_path)

            new_img = np.append(contrast_stretched_image, np.expand_dims(lbp, 2), axis=2)
            # new_img = np.append(new_img, np.expand_dims(lbp, 2), axis=2)
            new_tensor_img = transforms.ToTensor()(new_img)

            return new_tensor_img
        except Exception as e:
            print(f'Error {e} with image {img_path}')
            return None
    
    def _predict_big(self, img_path:str):
        img_tensor = self._transform_image(img_path)
        t = torch.zeros((1, 4, 512, 512), dtype=torch.float32)
        t[0] = img_tensor
        with torch.no_grad():
            mo_pred = torch.nn.functional.softmax(self.big_model(t), dim = 1).numpy() #mo(t).numpy()
        pred = mo_pred.tolist()
        return pred[0]
    
    
    def _get_top(self, pred_dist:list, n:int)->dict:
        '''Returns top-n most probable classes.'''
        pred_dict = dict(enumerate(pred_dist))
        c = Counter(pred_dict)
        top = c.most_common(n)
        return top

    def _convert_num_to_class(self, num:int, class_names:list[str])->str:
        '''Converts the number used by the model into the corresponding class name.'''
        return class_names[num]
    
    def save_orig_img_to_class_folder(self, img_path, img_class):
        try:
            print("Copying ", img_path, " to: ", os.path.join(self.classification_path, img_class[0]))
            if not os.path.exists(os.path.join(self.classification_path, img_class[0])):
                os.mkdir(os.path.join(self.classification_path, img_class[0]))
            shutil.copy(img_path, os.path.join(self.classification_path, img_class[0]))
        except Exception as e:
            print("An error occurred while copying to class folder...", e)
        return


    def get_classification_rank(self, img_path:str)->list:
        tr = transforms.Resize((512, 512))
        crop_img1 = tr.forward(Image.open(img_path))
        resized_img_path = 'resized_img_to_classify.jpeg'
        if crop_img1.mode in ("RGBA", "P"):
            crop_img1 = crop_img1.convert('RGB')
        crop_img1.save(resized_img_path)

        pred_dist = self._predict_big(resized_img_path)
        top3 = self._get_top(pred_dist, 1)
        top3_dist_classes = [(self._convert_num_to_class(num, self.class_names), p) for (num, p) in top3]
        
        self.save_orig_img_to_class_folder(img_path, top3_dist_classes[0])
        
        return top3_dist_classes
