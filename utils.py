import numpy as np
import scipy.misc as scm
import os
import tensorflow as tf

def get_img_list(img_list_dir):
    img_dir_list = []
    img_label_list = {}
    
    path = os.path.join(img_list_dir)
    file = open(path,'r')

    count = 0    
    for line in file:
        row = line.split('\n')[0].split()
        img_dir = row.pop(0)
        img_label = row.pop(0)
        #img_dir_list[count] = img_dir
        img_dir_list.append(img_dir)
        img_label_list[img_dir] = img_label
        count += 1
    
    file.close()
    return img_label_list, img_dir_list

def load_img(data_list, image_size):
    img = [get_img(img_path, image_size) for img_path in data_list]
    img = np.array(img)

    return img

def process_list(dir_list, num_list):
    output_list=[]
    for i in range(len(num_list)):
        output_list.append(dir_list[num_list[i]])
    return output_list

def get_img(img_path, data_size):
    img = scm.imread(img_path)
    #img = img[61:189,61:189,:]
    img_resize = scm.imresize(img,[data_size,data_size,3])
    #img_resize = img_resize/127.5 - 1.
    
    return img_resize

def preprocess_label(label, cls_num):
    label_out = []
    tmp = []    
    for i in range(len(label)):
        this_label = int(label[i]) - 1
        for j in range(cls_num):
            if j == this_label:
                tmp.append(1.)
            else:
                tmp.append(0.)
        label_out.append(tmp)
        tmp = []
    return label_out
