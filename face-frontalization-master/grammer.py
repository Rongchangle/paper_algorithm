import math
import cv2
import numpy as np
import os


def get_angle(rmat):
    u1,u2,u3 = (rmat[0][0],rmat[0][1],rmat[0][2])
    v1,v2,v3 = (rmat[1][0],rmat[1][1],rmat[1][2])
    w1,w2,w3 = (rmat[2][0],rmat[2][1],rmat[2][2])
    y = math.atan2(v3,w3)/math.pi*180  #x轴
    b = math.atan2(-u3,math.sqrt(v3*v3+w3*w3))/math.pi*180 #Y轴
    a = math.atan2(u2,u1)/math.pi*180 #Z轴
    return y,b,a

def get_angle2(rmat):
    x = math.atan2(rmat[2][1],rmat[2][2])/math.pi*180
    y = math.atan2(-rmat[2][0],math.sqrt(rmat[2][1]**2+rmat[2][2]**2))/math.pi*180
    z = math.atan2(rmat[1][0],rmat[0][0])/math.pi*180
    return x,y,z


