
import os
import sys
import numpy as np

from numpy.lib import utils

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import re

import cv2

from utils import face_det 
from utils import cv_utils

def get_all_list(root_path):
    print(root_path)

    with open(root_path + "AFAD-Full.txt", 'r') as f:
        lines = f.readlines()

    return lines

def do_afad():
    FaceDet = face_det.FaceDet(conf_threshold=0.4)

    root_path = "/home/hu/work/CV/dataset/afad/AFAD-Full/"
    
    image_save_root = "/home/hu/work/CV/dataset/afad/images/"
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)

    lines = get_all_list(root_path)

    fw = open("dataset/afad_list.txt", 'w')
    
    alpha = 8
    crop_image_size = 64

    cur_index = 0

    for line in lines:
        line = line.strip()   # ./15/111/109008-0.jpg
        info = line.split('/')[1:] # ['15', '111', '109008-0.jpg']

        cur_image_path = root_path + line[2:] # 15/111/109008-0.jpg
        
        image_src = cv_utils.decode_image(cur_image_path)
        if image_src is None:
            print("get none image->", cur_image_path)
            continue
        
        # image_draw = cv_utils.resize_image(image_src.copy())
        # d_h, d_w, _ = image_draw.shape

        s_h, s_w, _ = image_src.shape

        # (x0, y0, x1, y1, s) in src image
        face_boxs = FaceDet(frame=image_src.copy())

        # print(face_boxs, type(face_boxs), len(face_boxs))

        if len(face_boxs) != 1:
            face_boxs = np.array([[0, 0, s_w - 1, s_h - 1, 1.0]])
        
        for box in face_boxs:
            w = box[2] - box[0]
            h = box[3] - box[1]

            box[0] = max(box[0] - w / alpha, 0)
            box[1] = max(box[1] - h / alpha, 0)
            box[2] = min(box[2] + w / alpha, s_w - 1)
            box[3] = min(box[3] + h / alpha, s_h - 1)
            
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            cropped_im = image_src[y1:y2 + 1, x1:x2 + 1, :]
            resized_im = cv2.resize(cropped_im, (crop_image_size, crop_image_size), interpolation=cv2.INTER_LINEAR)

            cur_save_path = image_save_root + ("%s.jpg" % cur_index)

            age = info[0]
            gender = 0 # 0 -> fmale(女) 1 -> male(男)
            if '111' == info[1]:
                gender = 1

            fw.write(cur_save_path + '|score:1.00|age:%.2f|gender:%.2f\n' % (float(age), float(gender)))

            cv2.imwrite(cur_save_path, resized_im)

            # string = cur_save_path + '|score:1.00|age:%.2f|gender:%.2f\n' % (float(age), float(gender))
            # number = cv_utils.get_number_after_str(string=string, part='age')
            # print('转化为数字:',number)

            # x1 = int(box[0] / s_w * d_w)
            # y1 = int(box[1] / s_h * d_h)
            # x2 = int(box[2] / s_w * d_w)
            # y2 = int(box[3] / s_h * d_h)
        #     cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(image_draw, "face:%.2f" % (box[4]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # cv2.imshow("image_draw", image_draw)
        # cv2.waitKey(0)

        cur_index += 1

        if cur_index % 100 == 0:
            print("do afad %d in %d" % (cur_index, len(lines)))

    fw.close()
    print("do afad ok")

if __name__ == '__main__':

    do_afad()

    print("end process afad dataset !!!")