import shutil

import cv2
from skimage.transform import pyramid_gaussian
import torchvision.transforms as transforms
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import shuffle
from sklearn.model_selection import train_test_split



def resize_and_save_img(img_path:str, destination:str, num:int):
    '''Resizes image to 256x256 and saves new image in the destination.'''
    try:
        # dest = os.path.dirname(os.path.abspath(img_path))
        # if cv2.imread(img_path).shape[0] != 512:
            # print(img_path)
        tr = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        crop_img1 = Image.open(img_path)
        if crop_img1.height == 512 and crop_img1.width == 512: return
        if crop_img1.mode != 'RGB':
            crop_img1 = crop_img1.convert('RGB')
        crop_img1 = tr.forward(crop_img1)
        # crop_img1.save(os.path.join(destination,  str(num)+".jpeg"))
        crop_img1.save(img_path)
        # name = os.path.join(destination, os.path.basename(img_path))
        # crop_img1.save(img_path)
        os.rename(img_path, os.path.join(destination,  str(num)+".jpeg"))
        return
    except Exception:
        print("Error with:", img_path)
        return
    

def resize_batch(source_dir:str, destination:str, num:int):
    '''Creates a new folder (destination) with cropped images from the source dir.'''
    imgs = os.listdir(source_dir)
    for i in range(len(imgs)):
        path = os.path.join(source_dir, imgs[i])
        if os.path.isfile(path):
            if imgs[i].endswith('jpeg') or imgs[i].endswith('png') or imgs[i].endswith('jpg'):
                resize_and_save_img(path, source_dir, num)
                num += 1
                print(num)
    return num


def create_data_dir(source_dir:str, destination:str, num:int):
    '''Takes a forlder with class folders and crops all the images in the class folders.'''
    dirs = os.listdir(source_dir)
    for i in range(len(dirs)):
        path = os.path.join(source_dir, dirs[i])
        # dest =  os.path.join(destination, dirs[i])
        # new_name = dirs[i].replace('+', '').replace(',', '').replace(' ', '_')
        # new_name = ''.join(i for i in new_name if not i.isdigit())
        # if new_name[0] == '_': new_name = new_name[1:]
        # d = os.path.join(destination, new_name)
        # if not os.path.exists(d): os.mkdir(d)
        num = resize_batch(path, path, num)

    return num


def load_np_dataset(dataset_path:str)->tuple:
    '''Loads numpy dataset from .npz file.'''
    dataset = np.load(dataset_path)
    # print(dataset['images'].shape)
    # print(dataset['labels'].shape)
    return dataset['images'], dataset['labels']

def shuffle_data(dataset_path:str)->tuple:
    '''Shuffles numpy dataset loaded from .npz file and returns shuffled numpy arrays, images and labels.'''
    images, labels = load_np_dataset(dataset_path)
    inds = list(range(labels.shape[0]))
    shuffle(inds)
    images_shuffled = images[inds, :, :, :]
    labels_shuffeled = labels[inds,]

    return images_shuffled, labels_shuffeled

def create_train_test_set(images:np.ndarray, labels:np.ndarray)->tuple:
    '''Splits given images and labels into train and test datasets.'''
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, stratify=labels)
    return train_data, train_labels, test_data, test_labels

def split_test_within_one_class(class_name:str, class_path:str, test_folder:str, test_split:float):
    '''Splits images of one class into two sets, every 1/test_split image is
    saved in the new folder (test_class_path) and the rest of the images 
    stays in the original class folder.'''
    test_class_path = os.path.join(test_folder, class_name)
    if not os.path.exists(test_class_path):
        os.mkdir(test_class_path)
    
    imgs = os.listdir(class_path)
    # test_imgs_number = int(len(imgs)*test_split)

    for i in range(0, len(imgs), int(1/test_split)):
        old_name = os.path.join(class_path, imgs[i])
        new_name = os.path.join(test_class_path, imgs[i])
        os.rename(old_name, new_name)

    return

def create_test_folder(test_split:float, folder:str, test_folder:str):
    '''Splits a dataset in the given folder into two datasets according to the test_split. 
    Dataset in the folder becomes of size (1-test_split)\*original_dataset_size, 
    the result dataset of size test_split\*original_dataset_size is in the test_folder.'''
    class_folders = os.listdir(folder)
    for cf in class_folders:
        class_path = os.path.join(folder, cf)
        split_test_within_one_class(cf, class_path, test_folder, test_split)
    return


def rename(folder_path:str, dest:str):
    num = 4500
    dirs = os.listdir(folder_path)
    for d in dirs:
        # num = 2800
        d_path = os.path.join(folder_path, d)
        # d_path = folder_path
        imgs = os.listdir(d_path)
        for img in imgs:
            print(num)
            old_path = os.path.join(d_path, img)
            new_path = os.path.join(d_path, f'{str(num)}.jpeg')
            os.rename(old_path, new_path)
            num += 1
    return


def multiple_dirs_to_one(dir_path:str, metadata_path:str):
    dirs = os.listdir(dir_path)
    num = 0
    file_name = os.path.basename(dir_path)+'.txt'
    file_path = os.path.join(metadata_path, file_name)
    with open(file_path, 'w') as file:
        for d in dirs:
            d_path = os.path.join(dir_path, d)
            if os.path.isdir(d_path):
                imgs = os.listdir(d_path)
                for img in imgs:
                    file.write(f'{str(num)}, {d}\n')
                    old_path = os.path.join(d_path, img)
                    new_path = os.path.join(dir_path, f'{str(num)}.jpeg')
                    os.rename(old_path, new_path)
                    print(num)
                    num += 1
                if len(os.listdir(d_path)) == 0: os.rmdir(d_path)
    return


def clean_dataset(dataset_path:str):
    classes = os.listdir(dataset_path)
    for cl in classes:
        print(cl)
        class_path = os.path.join(dataset_path, cl)
        imgs = os.listdir(class_path)
        for img in imgs:
            img_path = os.path.join(class_path, img)

            try:
                im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            except:
                print('ERROR, DELETING IMG')
                os.remove(img_path)
    return


def augment(path):
    image = np.array(Image.open(path))
    if len(image.shape) == 2:
        image = Image.open(path)
        image = np.array(Image.merge("RGB", (image, image, image)))              
    h, w, _ = image.shape

    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    # image = tf.image.random_crop(image, size=[int(np.random.uniform(0.5, 1)*h), int(np.random.uniform(0.5, 1)*w), 3])
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, 0.2, 1)
    image = tf.image.random_jpeg_quality(image, 75, 100)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = image*255.0
    image = Image.fromarray(np.array(image, np.uint8))
    return image

def augment_class(class_path:str):
    imgs = os.listdir(class_path)
    num = 124

    for img in imgs:
        img_path = os.path.join(class_path, img)
        augmented_img = augment(img_path)
        augmented_img.save(os.path.join(class_path, str(num)+".jpeg"))
        num += 1
    return


def concat_imgs(imgs_path, dest):
    imgs = os.listdir(imgs_path)
    nums = []
    for img in imgs:
        img_num = img.split('_')[0]
        if img_num not in nums:
            img1 = Image.open(os.path.join(imgs_path, img_num+'_0.jpeg'))
            img2 = Image.open(os.path.join(imgs_path, img_num+'_1.jpeg'))
            new_im = Image.new('RGB', (512, 256))
            new_im.paste(img1, (0,0))
            new_im.paste(img2, (256,0))
            new_im.save(os.path.join(dest, img_num +'.jpeg'))
            nums.append(img_num)

    return


def move_class_images(class_path:str, dest:str):
    imgs = os.listdir(class_path)
    for img in imgs:
        old_path = os.path.join(class_path, img)
        new_path = os.path.join(dest, img)
        os.rename(old_path, new_path)
    return

def move_dist(source:str, dest:str):
    classes = os.listdir(source)
    for cl in classes:
        class_path = os.path.join(source, cl)
        dest_class_path = os.path.join(dest, cl)
        if not os.path.exists(dest_class_path):
            continue
        print(f'Moving images from {class_path} to {dest_class_path}')
        move_class_images(class_path, dest_class_path)
    return


import shutil
def copy_class_images(class_path:str, dest:str):
    imgs = os.listdir(class_path)
    for img in imgs:
        old_path = os.path.join(class_path, img)
        new_path = os.path.join(dest, img)
        shutil.copy(old_path, new_path)
    return


def copy_classes_to_one_class(source:str, dest:str):
    classes = os.listdir(source)
    for cl in classes:
        if cl == '60_adveritsement':
            continue
        print(f'Copying images from {cl}')
        class_path = os.path.join(source, cl)
        copy_class_images(class_path, dest)
    return
