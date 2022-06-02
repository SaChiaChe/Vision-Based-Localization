import os
import collections
import cv2
import numpy as np
import struct
import sys
import sqlite3
import glob
import pickle as pkl
from read_write_model import *
from database import *
from SuperPointDetectors import get_super_points_from_scenes_return

def Read_DB(method='Superpoint', database_path = './model/database.db'):
    # Open the database.
    print("Open database")
    db = COLMAPDatabase.connect(database_path)

    # For convenience, try creating all the tables upfront.
    print("Create tables")
    db.create_tables()

    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
#     print(camera_id)
#     print(model)
#     print(width)
#     print(height)
#     print(params)
#     print(prior)
#     print("")
    
    descriptors = None
    # Read and check keypoints and images.
    if method=="Superpoint":
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 4)))
            for image_id, data in db.execute(
                "SELECT image_id, data FROM keypoints"))
        images_name = dict((image_id, image_name)
                 for image_id, image_name in db.execute("SELECT image_id, name FROM images"))
    elif method=="SIFT":
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 6)))
            for image_id, data in db.execute(
                "SELECT image_id, data FROM keypoints"))
        images_name = dict((image_id, image_name)
                 for image_id, image_name in db.execute("SELECT image_id, name FROM images"))
        descriptors = dict(
            (image_id, blob_to_array(data, np.uint8, (-1, 128)))
            for image_id, data in db.execute(
                "SELECT image_id, data FROM descriptors"))
    
    return keypoints, images_name, descriptors

def SuperPoint_Ext(images_path = './images'):
    # get superpoint:
    return get_super_points_from_scenes_return(images_path)

def img_read(img_path):
    return
    

def create_New_DB(method, input_model_path, images_path, DB_path):
#     # get 3D model information
#     input_model_path = './model/'
    input_format = ".bin"
#     images_path = './images'
#     DB_path = './model/database.db'

    cameras, images, points3D = read_model(path=input_model_path, ext=input_format)

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))
    
    if method=='Superpoint':
        sps = SuperPoint_Ext(images_path)
        keypoints, images_name, descriptors = Read_DB(method='Superpoint', database_path = DB_path)
    elif method=='SIFT':
        keypoints, images_name, descriptors = Read_DB(method='SIFT', database_path = DB_path)
    elif method=="ORB":
        cv2.ORB_create()
        keypoints, images_name, descriptors = Read_DB(method='SIFT', database_path = DB_path)
        imgs = img_read(images_path)
    else:
        print("Unsupport method")
        return
    
    print("Create new database")
    if method=='Superpoint':
        print("Using Superpoint as feature extractor")
    elif method=='SIFT':
        print("Using SIFT as feature extractor")
    elif method=='ORB':
        print("Using ORB as feature extractor")
    # newDB element : [id, xyz, rgb, image_id, descriptor]
    newDB = []

    for i in sorted(points3D.keys()):
        point3D = points3D[i]
        des = []
        # for each image in image_ids
        for i in range(len(point3D[4])):
            if method=='Superpoint':
                des.append(sps[images_name[ point3D[4][i] ]]['descriptors'][ point3D[5][i] ])
            elif method=='SIFT':
                des.append(descriptors[point3D[4][i]][point3D[5][i]])
            elif method=='ORB':
                kp, descriptor = orb.compute(imgs[images_name[ point3D[4][i] ]], [keypoints[point3D[4][i]][point3D[5][i]]])
        des = np.asarray(des)
        des = np.mean(des, axis=0)
        newDB.append([point3D[0], point3D[1], point3D[2], point3D[4], des])

    with open('./model.pkl', 'wb') as f:
        pkl.dump(newDB, f)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify Colmap result to custom database")
    parser.add_argument("--method", choices=["SIFT", "Superpoint", "ORB"],
                        help="Feature extraction method Default:SIFT", default="SIFT")
    parser.add_argument("--input_model_path", help="Colmap Output Path Default:./model/", default="./model/")
    parser.add_argument("--images_path", help="Images data path Default:./images", default="./images")
    parser.add_argument("--DB_path", help="Database.db path Default:./model/database.db", default="./model/database.db")
    args = parser.parse_args()
    
    create_New_DB(method=args.method, input_model_path=args.input_model_path, images_path=args.images_path, DB_path=args.DB_path)