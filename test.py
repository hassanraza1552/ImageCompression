import numpy as np
import tensorflow as tf
from scipy import misc
#from PIL import Image
from os import listdir
import cv2
import matplotlib.pyplot as plt

def read_data(folderpath):
    list_of_files=listdir(folderpath)
    n=len(list_of_files)
    n=50000
    data=np.zeros(shape=(n,129,129,3),dtype=np.uint8)
    #print data.shape
    for x in range(n):
        de=misc.imread(folderpath+list_of_files[x])
        data[x]=de
    return data
d=read_data("/home/hassan/Thesis/tmp/")

np.save("/home/hassan/Thesis/data",d)


data=np.load("/home/hassan/Thesis/data.npy")

list_of_files=listdir("/home/hassan/Thesis/tmp")
for x in range(2):
    d=misc.imread("/home/hassan/Thesis/tmp/"+list_of_files[x])
    cv2.imshow('image',d)
    cv2.imshow('data',data[x])

    cv2.waitKey(0)

