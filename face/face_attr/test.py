import sys

from numpy.lib.npyio import save

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
import cv2
import glob
import yaml
import torch
import numpy as np

from utils import face_det
from utils import cv_utils
from symbols import *

def calculate_quality_score(text, file_name):
    #  [pretty, blur, glass, makeup, gender, mouthopen, smile, eyeopen, isface,block, yaw, pitch,age,mask]
    #  weight:   pretty:0.2; blur:0.1; makeup:0.1; smile:0.15; eyeopen:0.15; block:0.05; yaw:0.1; pitch:0.15;
    tiff = text.split('\n')
    numbers = [float(item.split(':')[1]) for item in tiff if ':' in item]
    #yaw>30: 1-abs(x)/90  yaw<30: -x^2/2700 +1
    numb_yaw = (-numbers[10]**2/2700+1) if numbers[10]<30 else (1-abs(numbers[10])/90)
    numb_pitch = (-numbers[10]**2/2700+1) if numbers[11]<30 else (1-abs(numbers[11])/90)

    score = numbers[0]*0.2 + (1-numbers[1])*0.1 + numbers[3]*0.1 + numbers[6]*0.15 + numbers[7]*0.15 + (1-numbers[9])*0.05 + numb_yaw*0.1 + numb_pitch*0.15

    return score

def test_face_attr(weight_path, config_file, image_path, log=None):
    # model
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    names = cfg['names']
    print(names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = get_mbf()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cv_utils.load_checkpoint(model, weight_path)
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112, device="cuda")
    torch.onnx.export(model, dummy_input, '/home/zhangkai/Myproject/face_attr/checkpoints/test.onnx',input_names = ['input'], 
                      output_names = ['pretty', 'blur', 'glass', 'makeup', 'gender', 'mouthopen', 'smile', 'eyeopen', 'isface', 'block', 'age', 'mask'])

    imgs = os.listdir(image_path)
    image_scores = {}
    for file_name in imgs:
        img = cv2.imread(os.path.join(image_path, file_name))
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        img = img[:, :, ::-1]   # BGR to RGB
        img = img/255
        img = (img-mean)/std
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.unsqueeze(0)
        
        face_attr = model(img)


        
        output_img = cv2.imread(os.path.join(image_path, file_name))
        output_img = cv2.resize(output_img, (500, 500))  

        num_columns = 1  
        column_width = output_img.shape[1] // num_columns
        text = ""
        for i, (x, y) in enumerate(zip(names, face_attr)):
            attr_str = f"{x}: {y.item():.5f}"
            if (i + 1) % num_columns == 0:
                attr_str += "\n"
            else:
                attr_str += " " * (column_width - len(attr_str))
            text += attr_str
        
        score = calculate_quality_score(text, file_name)
        image_scores[file_name] = score
        text = str('score: ') + str(round(score,5)) + "\n" + text 
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.6
        text_thickness = 2
        line_height = 30
        text_x = 10
        text_y = 30

        for i, line in enumerate(text.split('\n')):
            cv2.putText(output_img, line, (text_x, text_y + i * line_height), text_font, text_scale, (0, 0, 255), text_thickness)

        output_file_name = os.path.splitext(file_name)[0] + '_output.jpg'
        output_file_path = os.path.join('/home/zhangkai/Myproject/face_attr/output6', output_file_name)
        cv2.imwrite(output_file_path, output_img)

        if log is not None:
            log.write(file_name + '\n')
            log.write(text + '\n')

    sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)  # 根据分数降序排序
    top_images = sorted_images[:5]  # 获取分数最高的前n张图片
    print(sorted_images)
    output_best_folder = '/home/zhangkai/Myproject/face_attr/best_photo6'
    if not os.path.exists(output_best_folder):
        os.makedirs(output_best_folder)

    for image_name, score in top_images:
        image_best_path = os.path.join(image_path, image_name)
        output_best_path = os.path.join(output_best_folder, image_name)
        cv2.imwrite(output_best_path, cv2.imread(image_best_path))
    print("Saved image:", output_best_path)


if __name__ == "__main__":
    
    weight_path = '/home/zhangkai/Myproject/face_attr/checkpoints/sample6/last.pt'
    config_file = '/home/zhangkai/Myproject/face_attr/configs/face_attr.yaml'
    image_path = '/home/zhangkai/Myproject/face_attr/photo'
    log = open('log_test', 'w')

    test_face_attr(weight_path, config_file, image_path, log)

    print("end all test !!!")
