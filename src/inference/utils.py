import time
import numpy as np
import torch
import cv2 as cv
import matplotlib_inline
import glob
import open3d as o3d
import pickle as pkl
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as sciR

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

# Open3d initialization
def Init_o3d():
    # initialize visual window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=720)
    vis.get_render_option().load_from_json("./renderoption.json")
    return vis

# Draw 3D model
def Gen3DModel(database):
    points = np.asarray([pt[1] for pt in database])
    colors = np.asarray([pt[2] for pt in database])
    colors = colors/255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def draw_3dmodel(useSparse, DB, vis, model=None):
    ## draw 3D Model
    if useSparse:
        Model3DpointsSet = Gen3DModel(DB)
        vis.add_geometry(Model3DpointsSet)
    else:
        vis.add_geometry(model)
    # draw on o3d
    vis.poll_events()
    
# Pyramid
# Add new position
def add_new_pyramid(Intr, RT, vis, color=(1, 0, 0)):
    pointLineSet = o3d.geometry.LineSet.create_camera_visualization(Intr, RT, scale=0.3)
    pointLineSet.paint_uniform_color(color)
    vis.add_geometry(pointLineSet)
    return

def add_match_lineset(point_from, point_to_list, inlier):
    colors = np.array([[0,0,1], [0,1,0]])
    temp = np.zeros(len(point_to_list), dtype=bool)
    temp[inlier.flatten()] = True
    colors = colors[temp.astype(int)]
    num_line = len(point_to_list)
    pts = [point_from, *point_to_list]
    line_indices = list(np.concatenate(
        (
            np.zeros((num_line, 1), dtype=int), 
            np.arange(1, num_line+1, 1, dtype=int)[:, np.newaxis]
        ), 
        axis=1,
    ))
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(pts)
    lineset.lines = o3d.utility.Vector2iVector(line_indices)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    #     lineset.paint_uniform_color((0,0,1))

    return lineset

def R_T_to_invRT(Rv, T):
    temp_RT = np.zeros((4,4), np.float64)
    temp_RT[3,3] = 1
    RM = sciR.from_rotvec(Rv.squeeze()).as_matrix()
    temp_RT[0:3, 0:3] = RM
    temp_RT[0:3, 3] = T.T
    return temp_RT

class timer():
    def __init__(self):
        self.st = 0
        self.et = 0

    def tick(self):
        self.st = time.time()

    def tock(self):
        self.et = time.time()

    def get_time(self):
        return self.et - self.st

def write_img(img, texts, colors, offset=[10, 30]):
    for i in range(len(texts)):
        text = texts[i]
        color = colors[i]
        org = offset.copy()
        org[1] += 35*i
        cv.putText(img=img, text=text, org=tuple(org), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=BLACK, thickness=10)
        cv.putText(img=img, text=text, org=tuple(org), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=color, thickness=2)
