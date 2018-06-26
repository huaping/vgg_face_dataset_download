# -*- coding: UTF-8 -*-
#!/usr/bin/python
# History
# 06.26.2018 18:37:38  加入多线程下载
#Download the VGG face dataset from URLs given by http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz
from scipy import misc
import numpy as np
from skimage import io
import time
import os
import socket
from urllib2 import HTTPError, URLError
from httplib import HTTPException
import time
import Queue
import threading
import glob

datasetDescriptor = '/home/alex/tmp/DL/data/vgg/vgg_face_dataset/files'
resultPath = '/home/alex/tmp/DL/data/vgg/vgg_face_dataset/faces'

def saveErrorMessageFile(fileName, errorMessage):
  #print(errorMessage)
  with open(fileName, "w") as textFile:
    textFile.write(errorMessage)

def toRgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

def process_url(name, line):
    socket.setdefaulttimeout(15)
    dirName = name
    x = line.split(' ')
    fileName = x[0]   #第一列，文件名
    url = x[1]   #第二列，url
    box = np.rint(np.array(map(float, x[2:6])))  # x1,y1,x2,y2  #第3列到6，为人脸坐标
    imagePath = os.path.join(resultPath, dirName, fileName+'.png')
    classPath = os.path.join(resultPath, dirName)   #目标文件夹为人名
    if not os.path.exists(classPath):
        os.makedirs(classPath)
    if not os.path.exists(imagePath):
        try:
            img = io.imread(url)   #加载图片到img
        except (HTTPException, HTTPError, URLError, IOError, ValueError, IndexError, OSError) as e:
            errorMessage = '{}: {}'.format(url, e)
            print(errorMessage)
        else:
            try:
              #if image is gray and then translate to rgb
              if img.ndim == 2:
                img = toRgb(img)
              if img.ndim != 3:
                raise ValueError('Wrong number of image dimensions')
              hist = np.histogram(img, 255, density=True)
              if hist[0][0]>0.9 and hist[0][254]>0.9:
                raise ValueError('Image is mainly black or white')
              else:
                # Crop image according to dataset descriptor
                box=box.astype('int')
                imgCropped = img[box[1]:box[3],box[0]:box[2],:]
                # Scale to 256x256
                imgResized = misc.imresize(imgCropped, (256,256))
                # Save image as .png
                print(imagePath)
                misc.imsave(imagePath, imgResized)
            except ValueError as e:
              errorMessage = '{}: {}'.format(url, e)
              print(errorMessage)

def produce_urls(dir_path):
    all_lines = []   # (name, line
    names = []
    for fpath in glob.glob(os.path.join(datasetDescriptor, '*.txt')):
        name = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, 'rt') as f:
            tmp_lines = f.readlines()
            all_lines.extend(tmp_lines)
            for line in tmp_lines:
                names.append(name)
    return names, all_lines

queue = Queue.Queue()
class ThreadAddingFace(threading.Thread):
    def __init__(self,queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            #从队列中获取name,line
            name, line = self.queue.get()
            #print(name)  #
            try:
                #print(add_face(file_path, random_name()))
                #print(name,line)
                process_url(name, line)
            except:
                print("exception")
            #通知队列任务完成
            self.queue.task_done()

def multi_process_downloading():
    start = time.time()
    #10 threads
    for i in range(7):
        t = ThreadAddingFace(queue)
        t.setDaemon(True)
        t.start()
    #向队列中填充数据
    names, all_lines = produce_urls(datasetDescriptor)
    index = 0
    for line in all_lines:
        queue.put((names[index], line))
        index=index+1
    queue.join()
    print ("花费时间: %s seconds" % (time.time() - start))


if __name__ == '__main__':
  multi_process_downloading()
