import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import datasets, transforms
from skimage import feature

def my_loader(f:str)->Image.Image:
    img = Image.open(f)
    return img.convert("RGB")

MY_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

radius = 3
n_points = radius * 8

class my_dataset(datasets.DatasetFolder):
    def __init__(self, root_dir):
        super().__init__(root=root_dir, loader=my_loader, extensions=MY_IMG_EXTENSIONS)
        self.classes = self.classes
        self.class_to_idx = self.class_to_idx
        self.imgs = self.samples

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
    
    def _extract_edges(self, img_path:str):
        '''Extracts edges of the given image and returns them.'''
        gray_img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)# for AlexNet models, ensemble
        blur = cv2.GaussianBlur(gray_img, (3,3), 0)
        edges = cv2.Canny(blur, threshold1=100, threshold2=200)

        return edges

    def _get_fft(self, img_path:str):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return img_back.astype(np.float32)

    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        img_path, label = self.imgs[index]
        # print(img_path)
        contrast_stretched_image = self._stretch_contrast(img_path)
        # fft = self._get_fft(img_path)
        lbp = self._extract_lbp(img_path)
        # edges = self._extract_edges(img_path)

        new_img = np.append(contrast_stretched_image, np.expand_dims(lbp, 2), axis=2)
        # new_img = np.append(new_img, np.expand_dims(lbp, 2), axis=2)
        new_tensor_img = transforms.ToTensor()(new_img)

        return new_tensor_img, label