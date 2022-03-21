import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import paddle
from matplotlib.image import imread
import math
import cv2
import os
import dlib
from PIL import Image, ImageFont
import paddlehub as hub
import threading
from test_model import Facenet


def become_160(filename):
    '''
    将filename识别出的人脸框处理成160*160大小
    输入：
    filename：图片地址

    输出：
    new_image：将检测到的人脸框按比例缩放到160*160的灰色图中
    r：人脸框的信息
    '''
    print(type(filename))
    if isinstance(filename, str):
        image = cv2.imread(filename)
    elif isinstance(filename, np.ndarray):
        image = filename
    module = hub.Module(name="pyramidbox_lite_mobile_mask")
    try:
        results = module.face_detection(images=[image], use_multi_scale=True, shrink=0.6, visualization=False)
    # 若没检测到人脸
    except ValueError:
        # 考虑到批量喂数据，在此将批量数据中不合格的数据删除
        # os.remove(filename)
        print('此图片未检测出人脸')
        return
    # 转换颜色通道
    image=image[:,:,::-1]
    for result in results:
        # 选取目标检测result中的人脸框信息
        r=result['data']
        a=np.array([],dtype="int64")
        #左上x和y
        a=np.append(a,r[0]['left'].astype(np.int64))
        a=np.append(a,r[0]['top'].astype(np.int64))
        #右下x和y
        a=np.append(a,r[0]['right'].astype(np.int64))
        a=np.append(a,r[0]['bottom'].astype(np.int64))
    bboxs=a
    # 根据预测框裁剪图像
    crop_img = np.array(image)[int(bboxs[1]):int(bboxs[3]), int(bboxs[0]):int(bboxs[2])]
    image = crop_img
    #保证图像长宽比不变将图片缩放160*160
    new_image=letterbox_image(np.uint8(image), (160,160))
    #返回
    return new_image,r

def letterbox_image(image, size):
    '''
    将图片按照长宽比例不变缩放到指定大小的灰色图中
    输入：
    image：numpy.addry格式图像
    size：指定大小 (w,h)
    '''
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh), Image.BICUBIC)
    new_image = np.ones([size[1],size[0],3])*128
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image

def pridect(filename):
    '''
    传入一张图片将人脸信息生成128维向量
    输入：
    filename：图片路径
    输出：
    img1.numpy()：人脸信息的128维向量
    r：人脸框相关信息
    '''
    model=Facenet()
    #传入模型
    params_dict = paddle.load('palm_new.pdparams')
    model.set_state_dict(params_dict)
    #设置模型为预测模式
    model.eval()
    #将图片格式化为160*160且进行归一化

    img_crop,r=become_160(filename)
    img_crop=img_crop/255
    #将图片通道放在第一个维度
    img_crop = np.transpose(img_crop, (2, 0, 1))
    #添加一个维度，转换成神经网络标准输入格式[N,C,H,W]
    img_crop=np.expand_dims(img_crop,axis=0)
    #转换为tensor喂入神经网络
    img_crop=paddle.to_tensor(img_crop,dtype='float32')
    #预测
    img1=model(img_crop)
    return img1.numpy(),r

def crop_ori_pic(filename,savename):
    '''
    将输入的图片裁剪成500*500尺寸
    输入：
    filename：待裁剪图片地址
    输出：
    savename：裁剪后图片的存放地址
    '''
    img=cv2.imread(filename)
    img=img[:,:,::-1]
    new_image=letterbox_image(np.uint8(img), (500,500))
    im = Image.fromarray(np.uint8(new_image))
    im.save(savename)

#裁剪5张图片，前三张为不戴口罩的照片，后三张为佩戴口罩的照片，图片文件均在face_test中
# crop_ori_pic('data/face_test/no_mask/wufan.jpg','data/face_test/no_mask_crop/wf_crop.jpg')
# crop_ori_pic('data/face_test/no_mask/zhangmengmeng.jpg','data/face_test/no_mask_crop/zmm_crop.jpg')
# crop_ori_pic('data/face_test/no_mask/zxh.jpg','data/face_test/no_mask_crop/zxh_crop.jpg')
# crop_ori_pic('data/face_test/mask/wufanmask.jpg','data/face_test/mask_crop/wfmask_crop.jpg')
# crop_ori_pic('data/face_test/mask/zmmmask.jpg','data/face_test/mask_crop/zmmmask_crop.jpg')

# crop_ori_pic('data/face_test/mask/ltlmask.jpg','data/face_test/mask_crop/ltlmask_crop.jpg')
crop_ori_pic('data/face_test/no_mask/liutinglong.jpg','data/face_test/no_mask_crop/ltl_crop.jpg')

def compare_faces_ordered(encodings, face_names, encoding_to_check):
    '''
    将编码列表与要检查的候选对象进行比较时，返回有序的距离和名称
    '''

    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))


def compare_faces(encodings, encoding_to_check):
    '''
    返回编码列表与要检查对象的欧式距离列表
    '''
    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))




def process(stream):
    # 将图片放入通过facenet产生[1,128]维向量，取第0维度

    t1 = threading.Thread(target=queryframe(stream), daemon=True, args=())
    t1.start()
def queryframe(stream):
    fontStyle = ImageFont.truetype(
        "simsun.ttc", 30, encoding="utf-8")
    zxh_encoding, r = pridect('data/face_test/no_mask_crop/zxh_crop.jpg')
    zxh_encoding = zxh_encoding[0]
    wf_encoding, r = pridect('data/face_test/no_mask_crop/wf_crop.jpg')
    wf_encoding = wf_encoding[0]
    zmm_encoding, r = pridect('data/face_test/no_mask_crop/zmm_crop.jpg')
    zmm_encoding = zmm_encoding[0]
    ltl_encoding, r = pridect('data/face_test/no_mask_crop/ltl_crop.jpg')
    ltl_encoding = ltl_encoding[0]
    # 姓名列表以及对应的特征向量列表
    names = ["liutinglong", "wf", "zmm", "zxh"]
    known_encodings = [zxh_encoding, wf_encoding, zmm_encoding, ltl_encoding]
    while stream.isOpened():
        flag, img_rd = stream.read()  # Get camera video stream
        #cv2.imshow('frame', img_rd)
        if flag:
            kk = cv2.waitKey(1)
            # 编码待预测口罩人脸
            unknown_name = img_rd
            unknown_encoding, results = pridect(unknown_name)
            unknown_encoding = unknown_encoding[0]
            # unknown_image = cv2.imread(unknown_name)
            unknown_image = img_rd
            # 对比人脸
            computed_distances = compare_faces(known_encodings, unknown_encoding)
            computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)

            # 返回对比信息
            print(computed_distances)
            print(computed_distances_ordered)
            print(ordered_names)

            # 绘制预测时产生的人脸框
            for result in results:
                a = np.array([], dtype="int64")
                a = np.append(a, result['left'].astype(np.int64))
                a = np.append(a, result['top'].astype(np.int64))
                a = np.append(a, result['right'].astype(np.int64))
                a = np.append(a, result['bottom'].astype(np.int64))
            # 画矩形框 距离靠左靠上的位置
            pt1 = (a[0], a[1])  # 左边，上边   #
            pt2 = (a[2], a[3])  # 右边，下边
            cv2.rectangle(unknown_image, pt1, pt2, (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            imgzi = cv2.putText(unknown_image, result['label'] + '-' + ordered_names[0], (int(a[0]), int(a[1])), font, 1,
                                (0, 255, 255), 4)
            # 图像，          文字内容，                             坐标(右上角坐标)，      字体，大小，  颜色，   字体厚度
            # 将结果保存在result_m.jpg文件夹中

            #cv2.imwrite('result_wf.jpg', unknown_image)

            if kk == ord('q'):
                break
            cv2.imshow('Intelligent community face detection platform', imgzi)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Get video stream from camera
    process(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':

    main()