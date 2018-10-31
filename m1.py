import numpy as np
from Utility import *
import tensorflow as tf
from scipy import misc
#from PIL import Image
from os import listdir
#import cv2
import matplotlib.pyplot as plt
data=np.load("/local_scratch1/hassan/data_256x256.npy")



data=data[:500]
data_tensor=tf.constant(data,dtype=tf.float16,name="dataTensor")


def loss_psnr(im1,ref):
    psnr = tf.image.psnr(im1,ref,max_val=255,name="PSNR")
    #loss=tf.add(psnr,ssim)
    return psnr
def loss_ms_ssim(im1,ref):
    ms_ssim = tf.image.ssim_multiscale(im1,ref,max_val=255,power_factors=[0.25,0.25,0.25,0.25])
    return ms_ssim
def loss_ssim(im1,ref):
    ssim = tf.image.ssim(im1,ref,max_val=255)
    return ssim

def all_loss(im1,ref):
    return tf.add(tf.add(loss_psnr(im1,ref),loss_ms_ssim(im1,ref)),loss_ssim(im1,ref))




weights={
    "en1" : tf.get_variable("ENWC1",shape=[5,5,3,64],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "en2" : tf.get_variable("ENWC2",shape=[5,5,32,16],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "en3" : tf.get_variable("ENWC3",shape=[5,5,8,4],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "en4" : tf.get_variable("ENWC4",shape=[5,5,2,1],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),

    "de1" : tf.get_variable("DEWC1",shape=[5,5,1,2],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "de2" : tf.get_variable("DEWC2",shape=[5,5,4,8],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "de3" : tf.get_variable("DEWC3",shape=[5,5,16,32],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "de4" : tf.get_variable("DEWC4",shape=[5,5,64,3],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),

    "tcon1" : tf.get_variable("DEWTC1",shape=[5,5,2,4],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "tcon2" : tf.get_variable("DEWTC2",shape=[5,5,8,16],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),
    "tcon3" : tf.get_variable("DEWTC3",shape=[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)
   # "tcon4" : tf.get_variable("DEWTC4",shape=[5,5,12,4],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16),

   

}
bias={
    "eb1" : tf.Variable(tf.random_normal([64],dtype=tf.float16)),
    "eb2" : tf.Variable(tf.random_normal([16],dtype=tf.float16)),
    "eb3" : tf.Variable(tf.random_normal([4],dtype=tf.float16)),
    "eb4" : tf.Variable(tf.random_normal([1],dtype=tf.float16)),

    "db1" : tf.Variable(tf.random_normal([8],dtype=tf.float16)),
    "db2" : tf.Variable(tf.random_normal([16],dtype=tf.float16)),
    "db3" : tf.Variable(tf.random_normal([32],dtype=tf.float16)),
    "db4" : tf.Variable(tf.random_normal([3],dtype=tf.float16)),
}

def loss(im1,ref):
    psnr = tf.image.psnr(im1,ref,max_val=255,name="PSNR")
    ssim = tf.image.ssim_multiscale(im1,ref,max_val=255,power_factors=[0.25,0.25,0.25,0.25])
    loss = tf.add(psnr,ssim)
    return loss

input_data = tf.placeholder(dtype=tf.float16,shape=[None,256,256,3],name="input_data")
en_l1 = conv2d(input_data,weights["en1"],bias["eb1"])
max_l1=maxpool2d(en_l1,2)
en_l2 = conv2d(max_l1,weights["en2"],bias["eb2"])
max_l2=maxpool2d(en_l2,2)
en_l3 = conv2d(en_l2,weights["en3"],bias["eb3"])
max_l3=maxpool2d(en_l3,2)
en_l4 = conv2d(en_l3,weights["en4"],bias["eb4"])

de_l1 = conv2d(en_l4,weights["de1"],bias["db1"])
dims_de_1=de_l1.get_shape().as_list()
de_tcon_1=tf.nn.conv2d_transpose(value=de_l1,filter=weights["tcon1"],output_shape=(dims_de_1[0],dims_de_1[1]*2,dims_de_1[2]*2,dims_de_1[3]),strides=[1,2,2,1])
de_l2 = conv2d(de_tcon_1,weights["de2"],bias["db2"])
dims_de_2=de_l2.get_shape().as_list()
de_tcon_2=tf.nn.conv2d_transpose(value=de_l2,filter=weights["tcon2"],output_shape=(dims_de_2[0],dims_de_2[1]*2,dims_de_2[2]*2,dims_de_2[3]),strides=[1,2,2,1])
de_l3 = conv2d(de_tcon_2,weights["de3"],bias["db3"])

dims_de_3=de_l3.get_shape().as_list()
de_tcon_3=tf.nn.conv2d_transpose(value=de_l3,filter=weights["tcon3"],output_shape=(dims_de_3[0],dims_de_3[1]*2,dims_de_3[2]*2,dims_de_3[3]),strides=[1,2,2,1])
de_l4 = conv2d(de_tcon_3,weights["de4"],bias["db4"])
loss = all_loss(de_l4,data_tensor)


init = tf.global_variables_initializer()
#os.environ['CUDA_VISIBLE_DEVICES']='0'
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
#config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, device_count = {'GPU':0})
with tf.device('/gpu:0'):
	with tf.Session() as sess:
	    sess.run(init)
	    loss=sess.run(loss,feed_dict={input_data:data})
	    print loss[:50]
