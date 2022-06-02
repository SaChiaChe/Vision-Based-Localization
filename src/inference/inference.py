import os
import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
from utils import *
from scipy.spatial.transform import Rotation

from detector.detector import FeatureDetector
from matcher.matcher import FeatureMatcher
from solver.solver import PoseSolver

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

class SimpleVO:
    def __init__(self, args):
        # args: input, camera_parameters, useSparse, sparse_model_file, dense_model_file
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        self.feature_type = args.feature_type
        self.match_method = args.match_method
        self.calculation_method = args.calculation_method
        self.use_ransac = args.use_ransac
        self.resize = args.resize
        self.view_scale = args.view_scale
        self.save_dir = args.save_dir
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*'))), key=lambda x: int(os.path.split(x)[-1].split('_')[-1][:-4]))
        self.useSparse = args.useSparse #False
        self.model = None
        if not self.useSparse:
            model_path = args.dense_model_file # './fused.ply'
            sample_ratio = 0.01
            self.model = o3d.io.read_point_cloud(model_path)
            self.model = self.model.random_down_sample(sample_ratio)
        self.DB = pkl.load(open(args.sparse_model_file, 'rb'))
        self.database_des = np.array([s[4] for s in self.DB], dtype=np.float32)
        self.database_xyz = np.array([s[1] for s in self.DB], dtype=np.float32)

        if args.init_rvec is None:
            self.position = None
        else:
            self.position = {'r': np.array(args.init_rvec), 't': np.array(args.init_tvec)}

        self.clock = timer()

    def run(self):
        vis = Init_o3d()
        view_control = vis.get_view_control()
        # coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # vis.add_geometry(coor_frame)

        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        draw_3dmodel(self.useSparse, self.DB, vis, self.model)

        self.intr = o3d.camera.PinholeCameraIntrinsic()
        self.intr.intrinsic_matrix = self.K
        self.intr.width = self.resize[0]
        self.intr.height = self.resize[1]
        
        keep_running = True
        lineset = None
        while keep_running:
            try:
                R, t, matches, inlier, frame_name, end = queue.get(block=False)
                if R is not None:
                    if end: break
                    #TODO:
                    if lineset is not None:
                        vis.remove_geometry(lineset)
                    # insert new camera pose here using vis.add_geometry()
                    RT = R_T_to_invRT(R, t)
                    point_from = ((np.linalg.inv(RT)[:3,3]).flatten())
                    lineset = add_match_lineset(point_from, self.database_xyz[matches], inlier)
                    vis.add_geometry(lineset)
                    add_new_pyramid(self.intr, RT, vis)

                    view_control.set_lookat(point_from)
                    view_control.set_front((1e-10, -1, 0))
                    invRT = np.linalg.inv(RT)
                    front_normal = (invRT @ np.array([0., 0., 1., 0.]))[:3]
                    view_control.set_up(front_normal)
                    view_control.scale(self.view_scale)
                    vis.capture_screen_image(os.path.join(self.save_dir, f'vis_{frame_name}.png'), True)

            except Exception as e:
                pass

            keep_running = keep_running and vis.poll_events()
            
        vis.destroy_window()
        p.join()

    def tocktick(self):
        self.clock.tock()
        time = self.clock.get_time()
        self.clock.tick()
        return time

    def process_frames(self, queue):
        self.detector = FeatureDetector(self.feature_type)
        self.matcher = FeatureMatcher(self.match_method)
        self.solver = PoseSolver(self.calculation_method, self.K, self.dist, self.use_ransac)
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        Localization_success = False
        detect_time, filter_time, match_time, solve_time, total_time = 0, 0, 0, 0, 0
        track_time = []
        for frame_path in self.frame_paths:
            frame_name = os.path.split(frame_path)[-1][:-4]
            try:
                img = cv.imread(frame_path)
                if self.resize[0] > 0:
                    img = cv.resize(img, self.resize)
                #TODO: compute camera pose here
                self.clock.tick()
                info = self.detector(img)
                detect_time = self.tocktick()
                if self.position is None:
                    filter_time = 0.
                    matches = self.matcher(self.database_des, info["descriptors"])
                    match_time = self.tocktick()
                    rvec, tvec, _, inlier = self.solver(self.database_xyz[matches[:, 0]], info["keypoints"][matches[:, 1]])
                    solve_time = self.tocktick()
                else:
                    RT = R_T_to_invRT(self.position['r'], self.position['t'])
                    invRT = np.linalg.inv(RT)
                    front_normal = (invRT @ np.array([0., 0., 1., 0.]))[:3]
                    point_vecs = self.database_xyz - invRT[:3, 3].reshape(-1, 3)
                    point_vecs = point_vecs / np.linalg.norm(point_vecs, axis=1)[:, np.newaxis]
                    inner_prod = np.dot(point_vecs, front_normal)
                    filter_indice = np.argwhere(inner_prod > np.cos(np.pi / 6)).flatten()
                    filtered_database_des = self.database_des[filter_indice]
                    filter_time = self.tocktick()
                    matches = self.matcher(filtered_database_des, info["descriptors"])
                    matches[:, 0] = filter_indice[matches[:, 0]]
                    match_time = self.tocktick()
                    rvec, tvec, _, inlier = self.solver(self.database_xyz[matches[:, 0]], info["keypoints"][matches[:, 1]])
                    solve_time = self.tocktick()

                total_time = detect_time + filter_time + match_time + solve_time
                track_time.append(total_time)
                print("{frame_name} | D {dt:.5f}s | F {ft:.5f}s | M {mt:.5f}s | S {st:.5f}s | Total {tt:.5f}s ({fps:.5f} FPS)".format(
                       frame_name=frame_name,
                       dt=detect_time,
                       ft=filter_time,
                       mt=match_time,
                       st=solve_time,
                       tt=total_time,
                       fps=1./total_time), end='\033[K\r')
                # print(f"rvec: {rvec.flatten()}, tvec: {tvec.flatten()}")

                self.position = {
                    "r": rvec,
                    "t": tvec
                }

                R = rvec
                t = tvec
                queue.put((R, t, matches[:, 0], inlier, frame_name, False))
                Localization_success = True

            except Exception as e:
                print(e)
                self.position = None
                Localization_success = False
            try:
                keypoints = [cv.KeyPoint(info["keypoints"][x[1]][0], info["keypoints"][x[1]][1], size=1) for x in matches]
                cv.drawKeypoints(img, keypoints, img)
            except:
                pass
            if Localization_success:
                texts = [f"Detection: {detect_time:.5f}s",
                         f"Filtering: {filter_time:.5f}s",
                         f"Matching: {match_time:.5f}s",
                         f"Solve Pos: {solve_time:.5f}s",
                         f"Total time: {total_time:.5f}s ({1./total_time:.2f} FPS)"]
                colors = [WHITE, WHITE, WHITE, WHITE, WHITE]
            else:
                texts = ['Localization Failed']
                colors = [RED]
            write_img(img, texts, colors)
            cv.imwrite(os.path.join(self.save_dir, frame_name+'.png'), img)
            # img = cv.resize(img, (1080, 720))
            # cv.imshow('frame', img)
            if cv.waitKey(30) == 27: break

        queue.put((1, None, None, None, None, True))
        median_time = np.median(track_time)
        with open(os.path.join(self.save_dir, 'log.txt'), 'w+') as f:
            print(f'Median execution time: {median_time} ({1./median_time} FPS)\033[K')
            f.write(f'Median execution time: {median_time} ({1./median_time} FPS)\n')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', 
        default='./images/', 
        help='directory of sequential frames'
    )
    parser.add_argument(
        '--camera_parameters', 
        default='camera_parameter/camera_parameters.npy', 
        help='npy file of camera parameters'
    )
    parser.add_argument(
        '--useSparse', 
        action='store_true', 
        help='Use sparse or dense model for visualization'
    )
    parser.add_argument(
        '--sparse_model_file', 
        default='../Model_Compression/compressed_model_pca.pkl', 
        help='sparse 3D model file'
    )
    parser.add_argument(
        '--dense_model_file', 
        default='./fused.ply', 
        help='Dense 3D model file'
    )
    parser.add_argument(
        "--feature_type",
        type=FeatureDetector.FeatureType,
        default=FeatureDetector.FeatureType.SIFT,
        help="Type of features extracted from image, e.g. SuperPoint, SIFT, etc..."
    )
    parser.add_argument(
        "--match_method",
        type=FeatureMatcher.MatchMethod,
        default=FeatureMatcher.MatchMethod.KD_TREE,
        help="Method to match feature points, e.g. Brute Force, KD-Tree etc..."
    )
    parser.add_argument(
        "--calculation_method",
        type=PoseSolver.SolveMethod,
        default=PoseSolver.SolveMethod.PNP,
        help="Calculate method for solver"
    )
    parser.add_argument(
        "--use_ransac",
        action='store_true',
        help="Whether to use ransac calculation to solve the pose"
    )
    parser.add_argument(
        "--init_rvec",
        type=float,
        default=None,
        nargs=3,
        help="initial rotation vector"
        # 0 0.14 0.06
    )
    parser.add_argument(
        "--init_tvec",
        type=float,
        default=None,
        nargs=3,
        help="initial translation vector"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=[1920, 1080],
        nargs=2,
        help="resize inference image to target size"
    )

    parser.add_argument(
        "--view_scale",
        type=float,
        default=-20,
        help="scale of visuzalization"
    )
    args = parser.parse_args()

    if not os.path.isdir('video'):
        os.mkdir('video')

    save_dir = '{input}_{model}_{feature}_{match}_{calc}_{w}x{h}'.format(
                input=os.path.split(args.input)[-1],
                model=os.path.split(args.sparse_model_file)[-1][:-4],
                feature=args.feature_type,
                match=args.match_method,
                calc=args.calculation_method,
                w=args.resize[0],
                h=args.resize[1])
    save_dir = os.path.join('video', save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    args.save_dir = save_dir

    vo = SimpleVO(args)
    vo.run()

    # init position for model: campus_sp.pkl images: campus3
    # --init_rvec 0.05056287 -0.0922757 0.0504959 --init_tvec 1.86439232 0.1080795 -1.17792502 --view_scale -20

    # init position for model: campus1.pkl images: campus3
    # --init_rvec 0.01953391 -0.159423 0.03049893 --init_tvec 1.95370437 0.13115526 -1.2827208 --view_scale -30