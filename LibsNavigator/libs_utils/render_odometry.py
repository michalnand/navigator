import numpy
import cv2


class RenderOdometry:

    def __init__(self, output_file_name = None):
        
        self.output_file_name = output_file_name
        self.writer = None

       
    def render(self, frame, keypoints, trajectory, fps):

        height  = frame.shape[0]
        width   = frame.shape[1]
        

        result_im = numpy.zeros((height, width, 3), dtype=numpy.float32)

        result_im[:, :, 0] = frame[:, :, 0]/255.0
        result_im[:, :, 1] = frame[:, :, 1]/255.0
        result_im[:, :, 2] = frame[:, :, 2]/255.0
        
        
        for i in range(len(keypoints)):
            ay = keypoints[i, 1]
            ax = keypoints[i, 0]

            result_im = cv2.circle(result_im, (int(ax), int(ay)), 1, (0, 0, 1), -1)


        '''
        keypoints_all = keypoints_all.astype(int)
        #result_im[keypoints_all[:, 1], keypoints_all[:, 0], 1] = 1.0


        for i in range(len(keypoints_all)):
            ax = keypoints_all[i, 0]
            ay = keypoints_all[i, 1]
            

            result_im = cv2.circle(result_im, (int(ax), int(ay)), 3, (0, 1, 0), -1)
        '''
        
        size = 128
        result_traj = self._plot_trajectory(trajectory, size, size)

        result_im[0:size, 0:size] = result_traj
    
        text = "fps = " + str(int(fps))
        result_im = cv2.putText(result_im, text, (5, height - 20), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 1, 0), 2, cv2.LINE_AA)
    
        cv2.imshow("visual odometry camera", result_im)
        cv2.waitKey(1)

        if self.output_file_name is not None and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            self.writer = cv2.VideoWriter(self.output_file_name, fourcc, 25.0, (width, height)) 

        if self.writer is not None:
            tmp = (255*result_im).astype(numpy.uint8)
            self.writer.write(tmp)


    def _plot_trajectory(self, trajectory, height = 400, width = 400):
        trajectory = numpy.array(trajectory)
        scale = 2.1*numpy.max(abs(trajectory)) + 10e-10
        size  = min(width, height)

        trajectory_x_norm = size*(trajectory[:, 0]/scale) + width/2
        trajectory_y_norm = size*(trajectory[:, 1]/scale) + height/2

        #color blending, from blue to red
        ca = numpy.zeros((3, ))
        ca[0] = 1.0

        cb = numpy.zeros((3, ))
        cb[2] = 1.0

        result_im = numpy.zeros((height, width, 3), dtype=numpy.float32)


        for i in range(len(trajectory)):

            x = trajectory_x_norm[i]
            y = trajectory_y_norm[i]

            #color blend
            alpha = i/len(trajectory)
            c = (1 - alpha)*ca + alpha*cb

            result_im = cv2.circle(result_im, (int(x), int(y)), 1, c, -1)

        '''
        winname = "visual odometry path"
        cv2.namedWindow(winname)       
        cv2.moveWindow(winname, 100, 300) 
        cv2.imshow(winname, result_im)
        '''

        return result_im




