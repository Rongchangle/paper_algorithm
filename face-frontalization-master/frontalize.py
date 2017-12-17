__author__ = 'Douglas'

import scipy.io as scio
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})

def swap1(c):
    for i in range(len(c)):
        temp = -c[i][2]
        c[i][2] = c[i][1]
        c[i][1] = temp


def swap2(c):
    for i in range(len(c)):
        swap1(c[i])

class ThreeD_Model:
    def __init__(self, path, name):
        self.load_model(path, name)

    def load_model(self, path, name):
        model = scio.loadmat(path)[name]
        self.out_A = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3 cameraMatrix
        self.size_U = model['sizeU'][0, 0][0] #1x2    #估计是标准图片的大小 320*320
        self.model_TD = np.asarray(model['threedee'][0,0], dtype='float32') #68x3 标准3D模型68个特征点的坐标
        self.indbad = model['indbad'][0, 0]#0x1  不知道是干什么的,里面是空的
        self.ref_U = np.asarray(model['refU'][0,0]) #ref_U是320*320*3的,但是矩阵内部出现了负值,可能是320*320个点的三维坐标



        swap1(self.model_TD)
        swap2(self.ref_U)

#img是输入图片,大小250*250.
#proj_matrix是内置矩阵×外界矩阵,大小3*4.
#ref_U是320*320*3的,320*320个点(头像+背景)的三维坐标
#eyemask是320*320*3的,标记眼睛部分??
def frontalize(img, proj_matrix, ref_U, eyemask):

    ACC_CONST = 800
    img = img.astype('float32')

    bgind = np.sum(np.abs(ref_U), 2) == 0

    #bgind 320*320 ref_U合并后取TRUE,FALSE,除了（0,0,0）是True,其他都是False
    #eye 320*320*3

    # count the number of times each pixel in the query is accessed
    threedee = np.reshape(ref_U, (-1, 3), order='F').transpose()
    # threedee是3*102400的

    temp_proj = proj_matrix * np.vstack((threedee, np.ones((1, threedee.shape[1]))))
    #temp_proj: 3*102400 估计是320*320个点的校正后的2D图坐标,第三维就是个除数
    temp_proj2 = np.divide(temp_proj[0:2, :], np.tile(temp_proj[2, :], (2,1)))
    #temp_proj2 是2*102400的,是320*320个点使用proj_matrix校正后的2D坐标

    '''
    proj_matrix是根据标准3d特征点和输入图片(可能是侧脸)2d特征点计算得到
    本来标准3d点集合(人脸+背景)用这个校正,成为了有图片特点的2d坐标集合(就是temp_proj2)
    
    '''

    bad = np.logical_or(temp_proj2.min(axis=0) < 1, temp_proj2[1, :] > img.shape[0])
    bad = np.logical_or(bad, temp_proj2[0, :] > img.shape[1])
    bad = np.logical_or(bad, bgind.reshape((-1), order='F'))
    bad = np.asarray(bad).reshape((-1), order='F')
    #bad是个np的布尔数组,大小(102400,),它是一维的,如果是true,说明那个像素不在范围

    nonbadind = np.nonzero(bad == 0)[0]

    #就是本来320*320个点经过范围筛选,只有29697个点可以使用
    #bad == 0是bad数组取反
    #nonbadind保存的是bad数组中false的所有下标, 使用pic3,大小(29697,),下面很多大小29697都是使用了pic3而已
    temp_proj2 = temp_proj2[:, nonbadind]
    #这里的temp_proj2是2*29697的matrix, 取了原来temp_proj2中对应的nonbadind下标的,应该是2d图片中范围之内的像素

    # because python arrays are zero indexed
    temp_proj2 -= 1 #意思是temp_proj2里面的所有数字减去一,和之前bad数组那边相对应了
    ind = np.ravel_multi_index((np.asarray(temp_proj2[1, :].round(), dtype='int64'), np.asarray(temp_proj2[0, :].round(),
                                dtype='int64')), dims=img.shape[:-1], order='F')
    #round()是小数四舍五入变成最近整数的意思,但是注意小数部分为0.5的时候,不确定是往上面取,还是下面取
    #ind是个1×29697的数组,大概记录了temp_proj2的二维坐标,不过使用了一个数字来表示

    synth_frontal_acc = np.zeros(ref_U.shape[:-1])
    #synth_frontal_acc 320*320的全零??
    ind_frontal = np.arange(0, ref_U.shape[0]*ref_U.shape[1])
    ind_frontal = ind_frontal[nonbadind]
    #ind_frontal 貌似和 nonbadind一样的（类型,数值顺序),保存的是bad数组中false的所有下标, 大小(29697,)
    print('ind_frontal')
    print(ind_frontal)
    c, ic = np.unique(ind, return_inverse=True)

    #c数组相当于ind去重排序了,c记录了输入图像校正后二维像素在范围内的所有坐标,坐标用一个数字来表示,c大小是(28553,)

    bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
    #bin_edges大小是（28554,）貌似是前后是正负inf,中间每个数都取了c数组相邻的每两个数的平均数,都变成小数了
    #bin_edges的意义就是保存了区间点的值（区间点有28544个,区间有28543个)

    count, bin_edges = np.histogram(ind, bin_edges)
    #histogram的意思不是太明白,但是大概猜出这是求ind数组的分布,比如ind数组中在XX区间的数字有多少个 XX区间原来就是bin_edges那里搞出来的
    #countshape是（28553,0）的,意思是有28553个坐标不同的点,但是count中所有数的和还是29697

    synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
    #目前synth_frontal_acc是（102400,)的全0数组

    synth_frontal_acc[ind_frontal] = count[ic]
    #synth_frontal_acc本来是102400的全零数组, ind_frontal是bad中所有false的下标
    #ind_frontal是bad数组所有false的下标大小（29697,）
    #ic大小为（29697,）

    synth_frontal_acc = synth_frontal_acc.reshape((320, 320), order='F')
    # synth_frontal_acc现在成了校正后2D标准图的某个像素出现的次数,就是论文的v(q'),q'是某个像素
    synth_frontal_acc = cv2.GaussianBlur(synth_frontal_acc, (15, 15), 30., borderType=cv2.BORDER_REPLICATE)
    frontal_raw = np.zeros((102400, 3))


    frontal_raw[ind_frontal, :] = cv2.remap(img, temp_proj2[0, :].astype('float32'), temp_proj2[1, :].astype('float32'), cv2.INTER_LINEAR)

    
    frontal_raw = frontal_raw.reshape((320, 320, 3), order='F')

    # which side has more occlusions?
    midcolumn = np.round(ref_U.shape[1]/2)
    sumaccs = synth_frontal_acc.sum(axis=0)
    midcolumn = midcolumn.astype('int32')
    sum_left = sumaccs[0:midcolumn].sum()
    sum_right = sumaccs[midcolumn+1:].sum()
    sum_diff = sum_left - sum_right


    if np.abs(sum_diff) > ACC_CONST: # one side is ocluded
        ones = np.ones((ref_U.shape[0], midcolumn))
        zeros = np.zeros((ref_U.shape[0], midcolumn))
        if sum_diff > ACC_CONST: # left side of face has more occlusions
            weights = np.hstack((zeros, ones))
        else: # right side of face has more occlusions
            weights = np.hstack((ones, zeros))
        weights = cv2.GaussianBlur(weights, (33, 33), 60.5, borderType=cv2.BORDER_REPLICATE)

        # apply soft symmetry to use whatever parts are visible in ocluded side
        synth_frontal_acc /= synth_frontal_acc.max()

        weight_take_from_org = 1. / np.exp(0.5+synth_frontal_acc)
        weight_take_from_sym = 1 - weight_take_from_org

        weight_take_from_org = np.multiply(weight_take_from_org, np.fliplr(weights))
        weight_take_from_sym = np.multiply(weight_take_from_sym, np.fliplr(weights))

        weight_take_from_org = np.tile(weight_take_from_org.reshape(320, 320, 1), (1, 1, 3))
        weight_take_from_sym = np.tile(weight_take_from_sym.reshape(320, 320, 1), (1, 1, 3))

        weights = np.tile(weights.reshape(320, 320, 1), (1, 1, 3))

        denominator = weights + weight_take_from_org + weight_take_from_sym
        #权重矩阵,和得到的结果frontal_sym一一对应,到时候得到的矩阵所有值除以这个矩阵的对应就可以了
        frontal_sym = np.multiply(frontal_raw, weights) + np.multiply(frontal_raw, weight_take_from_org) + np.multiply(np.fliplr(frontal_raw), weight_take_from_sym)
        frontal_sym = np.divide(frontal_sym, denominator)

        # exclude eyes from symmetry

        frontal_sym = np.multiply(frontal_sym, 1-eyemask) + np.multiply(frontal_raw, eyemask)
        frontal_raw[frontal_raw > 255] = 255
        frontal_raw[frontal_raw < 0] = 0
        frontal_raw = frontal_raw.astype('uint8')
        frontal_sym[frontal_sym > 255] = 255
        frontal_sym[frontal_sym < 0] = 0
        frontal_sym = frontal_sym.astype('uint8')
    else: # both sides are occluded pretty much to the same extent -- do not use symmetry
        frontal_sym = frontal_raw
    return frontal_raw, frontal_sym