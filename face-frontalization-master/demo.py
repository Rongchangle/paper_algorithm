import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
import grammer

this_path = os.path.dirname(os.path.abspath(__file__))

#必须保证输入(image)是图片,否则直接报错

#如果没有检测人脸特征点,返回False, 1
#如果检测到,　返回True,处理后的图片
def demo(image):
    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    check.check_dlib_landmark_weights()
    # load detections performed by dlib library on 3D model and Reference Image
    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
    img = cv2.resize(image, (250, 250), interpolation=cv2.INTER_LINEAR)

    lmarks, f = feature_detection.get_landmarks(img)

    if not f :
        return False, 1

    # perform camera calibration according to the first face detected
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])

    print(proj_matrix)

    # load mask to exclude eyes from symmetry
    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    # perform frontalization

    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)

    '''
    plt.figure()
    plt.title('Image frontalized(Before symmetry)')
    plt.imshow(frontal_raw[:,:,::-1].astype('uint8'))
    plt.figure()
    plt.title('Image frontalized(After symmetry)')
    plt.imshow(frontal_sym[:,:,::-1].astype('uint8'))
    '''
    x, y, z = grammer.get_angle(rmat)
    print(('旋转的角度：　x: {}, y: {}, z: {}').format(x, y, z)) #估算大概的旋转角度
    image_output = frontal_sym.astype('uint8')
    return True, image_output



if __name__ == "__main__":
    #check.check_dlib_landmark_weights()
    # load detections performed by dlib library on 3D model and Reference Image
    #model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
    #model_td=model3D.model_TD
    image_input = cv2.imread("/home/rong/mypic/cropped/pic46cropped1.jpg", 1)
    flag, image_output = demo(image_input)
    if flag : #如果检测到特征点
        plt.imshow(image_output[:,:,::-1])
        plt.show()
    else:   #没有检测到特征点
        print('No face detected')

