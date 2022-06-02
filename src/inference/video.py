import cv2
import numpy as np
import glob
import os

view_size = (600, 400)
view_pos = (540, 50)
fps = 20

images_dir_name_list = glob.glob("./video/*")
temp = []
for i in images_dir_name_list:
    if os.path.isdir(i):
        temp.append(i)
images_dir_name_list = temp
video_output_path = './video/'
Good_rate = {}

for images_dir_name in images_dir_name_list:
    video_name = os.path.split(images_dir_name)[-1]
    print("Processing "+video_name)

    imgs_path_list = glob.glob(images_dir_name+'/*png')
    frames_path_list = [name for name in imgs_path_list if os.path.split(name)[-1].split('_')[0]!='vis']

    # Calculate relocalization rate
    frmNum = len(frames_path_list)
    viewNum = len(imgs_path_list) - len(frames_path_list)
    Good_rate[video_name] = viewNum*100 / frmNum
    
    video_frames = []
    pre_view = None
    size = None

    # read and combine frame and view
    for frame_path in frames_path_list:
        frame_name = os.path.split(frame_path)[-1]

        # check if view exist and read it
        view_path = os.path.join(images_dir_name, "vis_"+frame_name)
        view = None
        if os.path.exists(view_path):
            view = cv2.imread(view_path)
            view = cv2.resize(view, view_size)
            # update pre view
            pre_view = np.copy(view)
        else:
            view = pre_view
        
        # read frame
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (1920, 1080))
        height, width, layers = frame.shape
        size = (width,height)
        
        # combine frame and view(check if None)
        if view is not None:
            frame[ view_pos[0]:view_pos[0]+view_size[1] , view_pos[1]:view_pos[1]+view_size[0] ] = view

        # write to video frames
        video_frames.append(frame)

    output_video = cv2.VideoWriter(os.path.join(video_output_path, video_name+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(video_frames)):
        output_video.write(video_frames[i])
    output_video.release()

    print("Finish "+video_name)

# Write Good rate to txt file
log_name = './SucRate.txt'
with open(log_name, 'w') as f:
    for videoN in Good_rate:
        f.write("{:.3f}%  ".format(Good_rate[videoN]) + videoN)
print("Finish writing Good rate file")
