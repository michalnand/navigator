import cv2
import numpy
import time

from LibsNavigator import *


def load_images(path, count = 50, height = 512, width = 256):

    result = []
    for i in range(count):
        file_name = path + str(i).zfill(6) + ".png"
        img = cv2.imread(file_name)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if height is not None and width is not None:
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

        print("loading ", file_name, img.shape, img.dtype)

        result.append(img)

    return result


def load_poses(file_name):
    """
    Loads the GT poses

    Parameters
    ----------
    file_name (str): The file path to the poses file

    Returns
    -------
    poses (ndarray): The GT poses
    """
    poses = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            T = numpy.fromstring(line, dtype=numpy.float32, sep=' ')
            T = T.reshape(3, 4)
            T = numpy.vstack((T, [0, 0, 0, 1]))
            poses.append(T)

    return poses


def load_calib(file_name):
    """
    Loads the calibration of the camera
    Parameters
    ----------
    file_name (str): The file path to the camera file

    Returns
    -------
    K (ndarray): Intrinsic parameters
    P (ndarray): Projection matrix
    """
    with open(file_name, 'r') as f:
        params = numpy.fromstring(f.readline(), dtype=numpy.float32, sep=' ')
        P = numpy.reshape(params, (3, 4))
        K = P[0:3, 0:3]

    return K, P





if __name__ == "__main__":

    '''
    height_orig = 1080
    width_orig  = 1920
    
    height  = 1080//4
    width   = 1920//4 
    
    focal_y  = 1240*height/height_orig
    focal_x  = 1240*width/width_orig

    #initial position - identity matrix
    cur_pose     = numpy.eye(4, dtype=numpy.float32)
    
    #compute intrisics cameta matrix
    mat_k    = numpy.zeros((3, 3), dtype=numpy.float32)
    mat_p    = numpy.zeros((3, 4), dtype=numpy.float32)

    mat_k[0][2] = width/2
    mat_k[1][2] = height/2
    mat_k[0][0] = focal_x
    mat_k[1][1] = focal_y
    mat_k[2][2] = 1.0

    mat_p[0][2] = width/2
    mat_p[1][2] = height/2
    mat_p[0][0] = focal_x
    mat_p[1][1] = focal_y
    mat_p[2][2] = 1.0

    print(mat_k)
    print(mat_p)
    '''


    height_orig = 370
    width_orig  = 1226
    
    height  = 370 
    width   = 1226 
    
    focal_y  = 707*height/height_orig
    focal_x  = 707*width/width_orig

    count   = 1000
    path    = "/Users/michal/Movies/segmentation/kitty_odometry/03/"

    images  = load_images(path + "image_l/", count, height, width)


    cur_pose = numpy.eye(4, dtype=numpy.float32)
    cur_pose_inv = numpy.eye(4, dtype=numpy.float32)
    
    mat_k    = numpy.zeros((3, 3), dtype=numpy.float32)
    mat_p    = numpy.zeros((3, 4), dtype=numpy.float32)

    mat_k[0][2] = width/2
    mat_k[1][2] = height/2
    mat_k[0][0] = focal_x
    mat_k[1][1] = focal_y
    mat_k[2][2] = 1.0

    mat_p[0][2] = width/2
    mat_p[1][2] = height/2
    mat_p[0][0] = focal_x
    mat_p[1][1] = focal_y
    mat_p[2][2] = 1.0
    
    

    #cap = cv2.VideoCapture("/Users/michal/Movies/segmentation/park2.mp4")  


    vo  = VisualOdometry(mat_k, mat_p)


    trajectory = []

   

    time_prev = time.time() + 1
    time_now = time.time()
    fps = 0.0

    points = []

    render_odometry    = RenderOdometry()
    render_point_cloud = RenderPointCloud(height, width)

    step = 0

    '''
    while cap.isOpened():
    
        ret, frame = cap.read()

        if ret == True and step%5 == 0:
    '''


    for i in range(len(images)):

        frame = images[i]
        frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)

        q1, q2, q_3d, transf = vo.step(frame)
        
        cur_pose = numpy.matmul(cur_pose, numpy.linalg.inv(transf))


        x = cur_pose[0][3]
        y = cur_pose[2][3]
        z = cur_pose[1][3]
        
        trajectory.append([x, y])

        print("3D points = ", numpy.var(q_3d, axis=0))

        
        cur_pose_scaled = cur_pose.copy()

        
        scale = 0.01
        cur_pose_scaled[0][3]*= scale
        cur_pose_scaled[1][3]*= scale
        cur_pose_scaled[2][3]*= scale
        cur_pose_scaled[3][3]*= scale
        
        
        #points = []
        points.append(q_3d@cur_pose_scaled)


        k = 0.9
        time_prev   = time_now
        time_now    = time.time()
        fps = k*fps + (1.0 - k)*1.0/(time_now - time_prev)


        
        render_odometry.render(frame, q1, trajectory, fps)

        if step%30 == 0:
            render_point_cloud.render(points)

        step+= 1
            