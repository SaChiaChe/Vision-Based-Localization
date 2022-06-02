import numpy as np
import cv2
import pickle
import open3d as o3d
from sklearn.decomposition import PCA
import argparse

VERBOSE = False

def get_plane(p1, p2, p3):
    u = p2 - p1
    v = p3 - p1
    uxv = np.cross(u, v)
    if np.linalg.norm(uxv) < 1e-10:
        # if VERBOSE: print('3 points co-line')
        return None
    normal = uxv / np.linalg.norm(uxv)
    d = -np.dot(p1, normal)
    plane = np.concatenate((normal, [d]))
    return plane

def dis_point_plane(point, plane):
    point = np.hstack((point, np.ones(len(point)).reshape(-1, 1)))
    dis = np.abs(np.dot(point, plane)) / (np.linalg.norm(point, axis=1) + 1e-10)
    return dis

def dis_point_line(p, line):
    l1, l2 = line
    cross = np.cross(l1 - p, l2 - p)
    dis = np.linalg.norm(cross, axis=1) / (np.linalg.norm(l2-l1) + 1e-10)
    return dis

def plane_detection(points3d, inlier_prob=0.05, prob=0.99, patience=0.05, threshold=0.01):
    if len(points3d) < 3:
        return np.zeros(len(points3d), dtype=bool)
    best_support = 0
    best_inliers_idx = None
    best_std = np.inf
    epsilon = 1 - inlier_prob
    N = int(np.round(np.log(1-prob)/np.log(1-(1-epsilon)**3)))
    patience = int(np.round(N * patience))
    staleness = 0
    for i in range(N):
        # pick 3 random points
        rand_idx = np.random.choice(np.arange(len(points3d)), 3, replace=False)
        p3 = points3d[rand_idx]
        
        # get plane of the 3 points
        plane = get_plane(*p3)
        if plane is None: continue
        
        # get distance of points to plane
        dis = dis_point_plane(points3d, plane)
        
        # get inliers
        inliers_idx = dis < threshold
        inliers_dis = dis[inliers_idx]
        inliers = points3d[inliers_idx]
        inliers_std = np.std(inliers_dis)
        
        # check improve
        inlier_n = len(inliers)
        if (inlier_n > best_support) or ((inlier_n == best_support) and (inliers_std == best_std)):
            best_support = inlier_n
            best_inliers_idx = inliers_idx
            best_std = inliers_std
            staleness = 0
        
        staleness += 1
        if staleness > patience:
            break
    return best_inliers_idx

def plane_detection_PCA(points2d, inlier_prob=0.05, prob=0.99, patience=0.05, threshold=0.15):
    if len(points2d) < 3:
        return np.zeros(len(points2d), dtype=bool)
    best_support = 0
    best_inliers_idx = None
    best_std = np.inf
    epsilon = 1 - inlier_prob
    N = int(np.round(np.log(1-prob)/np.log(1-(1-epsilon)**3)))
    patience = int(np.round(N * patience))
    staleness = 0
    points3d = np.concatenate((points2d, np.zeros((points2d.shape[0],1))), axis=1)
    
    for i in range(N):
        # pick 2 random points
        rand_idx = np.random.choice(np.arange(len(points3d)), 2, replace=False)
        line = points3d[rand_idx]
        
        # get distance of points to line
        dis = dis_point_line(points3d, line)
        
        # get inliers
        inliers_idx = dis < threshold
        inliers_dis = dis[inliers_idx]
        inliers = points3d[inliers_idx]
        inliers_std = np.std(inliers_dis)
        
        # check improve
        inlier_n = len(inliers)
        if (inlier_n > best_support) or ((inlier_n == best_support) and (inliers_std == best_std)):
            best_support = inlier_n
            best_inliers_idx = inliers_idx
            best_std = inliers_std
            staleness = 0
        
        staleness += 1
        if staleness > patience:
            break
    return best_inliers_idx

def line_detection(points3d, inlier_prob=0.05, prob=0.99, patience=0.05, threshold=0.15):
    if len(points3d) < 2:
        return np.zeros(len(points3d), dtype=bool)
    best_support = 0
    best_inliers_idx = None
    best_std = np.inf
    epsilon = 1 - inlier_prob
    N = int(np.round(np.log(1-prob)/np.log(1-(1-epsilon)**2)))
    patience = int(np.round(N * patience))
    staleness = 0
    for i in range(N):
        # pick 2 random points
        rand_idx = np.random.choice(np.arange(len(points3d)), 2, replace=False)
        line = points3d[rand_idx]
        
        # get distance of points to line
        dis = dis_point_line(points3d, line)
        
        # get inliers
        inliers_idx = dis < threshold
        inliers_dis = dis[inliers_idx]
        inliers = points3d[inliers_idx]
        inliers_std = np.std(inliers_dis)
        
        # check improve
        inlier_n = len(inliers)
        if (inlier_n > best_support) or ((inlier_n == best_support) and (inliers_std == best_std)):
            best_support = inlier_n
            best_inliers_idx = inliers_idx
            best_std = inliers_std
            staleness = 0
        
        staleness += 1
        if staleness > patience:
            break
    return best_inliers_idx

def split_model(points3d, dx=1, dy=1, dz=1):
    xmin, ymin, zmin = np.min(points3d, axis=0)
    xmax, ymax, zmax = np.max(points3d, axis=0)
    
    xblocks = np.ceil((xmax - xmin) / dx).astype(int)
    yblocks = np.ceil((ymax - ymin) / dy).astype(int)
    zblocks = np.ceil((zmax - zmin) / dz).astype(int)
    
    px, py, pz = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    points_split = []
    for x in range(xblocks):
        for y in range(yblocks):
            for z in range(zblocks):
                xlower, xupper = xmin + x*dx, xmin + x*dx + dx
                ylower, yupper = ymin + y*dy, ymin + y*dy + dy
                zlower, zupper = zmin + z*dz, zmin + z*dz + dz
                
                x_id = (px >= xlower) * (px < xupper)
                y_id = (py >= ylower) * (py < yupper)
                z_id = (pz >= zlower) * (pz < zupper)
                
                block_id = x_id * y_id * z_id
                points_split.append(block_id)
    return points_split

def structure_detection(points3d, min_plane_points=500, min_line_points=50, method='vanilla', dxyz=None):
    if method.lower() == 'local':
        dx, dy, dz = dxyz
        points_split = split_model(points3d, dx=dx, dy=dy, dz=dz)
        plane_id = -np.ones(points3d.shape[0], dtype=int)
        line_id = -np.ones(points3d.shape[0], dtype=int)
        pid_shift = 0
        lid_shift = 0
        for block_id in points_split:
            if np.sum(block_id > 0) == 0: continue
            if VERBOSE: print("=============== New Block ===============")
            if VERBOSE: print(f'number of points: {np.sum(block_id > 0)}')
            pid, lid = structure_detection(points3d[block_id], min_plane_points=min_plane_points, min_line_points=min_line_points, method='vanilla')
            assert np.sum((pid >= 0) * (lid >= 0)) == 0
            shifted_pid = pid + (pid > 0) * pid_shift
            shifted_lid = lid + (lid > 0) * lid_shift
            plane_id[block_id] = shifted_pid
            line_id[block_id] = shifted_lid
            pid_shift += np.max(pid) + 1
            lid_shift += np.max(lid) + 1
        
        return plane_id, line_id
    
    else:
        # detect planes
        plane_id = -np.ones(points3d.shape[0], dtype=int)
        cur_id = 0
        remain_points = plane_id < 0
        if method.lower() == 'pca':
            pca = PCA(n_components=2)
        if VERBOSE: print('===== plane detection =====')
        if VERBOSE: print('id\tselected\tremained')
        while True:
            if method == 'pca' and cur_id > 0:
                if np.sum(remain_points) < min_plane_points:
                    if VERBOSE: print(f'Not enough remaining points: {np.sum(remain_points)}')
                    break
                inliers = plane_detection_PCA(pca.transform(points3d[remain_points]))
                inliers = np.where(remain_points)[0][inliers]
                if len(inliers) < min_plane_points:
                    if VERBOSE: print(f'Not enough support: {len(inliers)}')
                    break
                plane_id[inliers] = cur_id
                remain_points = plane_id < 0
                if VERBOSE: print(f'{cur_id}\t{len(inliers)}\t\t{np.sum(remain_points)}')
                cur_id += 1
            else: 
                if np.sum(remain_points) < min_plane_points:
                    if VERBOSE: print(f'Not enough remaining points: {np.sum(remain_points)}')
                    break
                inliers = plane_detection(points3d[remain_points])
                inliers = np.where(remain_points)[0][inliers]
                if len(inliers) < min_plane_points:
                    if VERBOSE: print(f'Not enough support: {len(inliers)}')
                    break
                plane_id[inliers] = cur_id
                remain_points = plane_id < 0
                if VERBOSE: print(f'{cur_id}\t{len(inliers)}\t\t{np.sum(remain_points)}')
                cur_id += 1
                if method == 'pca':
                    pca.fit(points3d[inliers])

        # detect lines
        line_id = -np.ones(points3d.shape[0], dtype=int)
        cur_id = 0
        remain_points = (line_id < 0) * (plane_id < 0)
        if VERBOSE: print('===== line detection =====')
        if VERBOSE: print('id\tselected\tremained')
        while True:
            if np.sum(remain_points) < min_line_points:
                if VERBOSE: print(f'Not enough remaining points: {np.sum(remain_points)}')
                break
            inliers = line_detection(points3d[remain_points])
            inliers = np.where(remain_points)[0][inliers]
            if len(inliers) < min_line_points:
                if VERBOSE: print(f'Not enough support: {len(inliers)}')
                break
            line_id[inliers] = cur_id
            remain_points = (line_id < 0) * (plane_id < 0)
            if VERBOSE: print(f'{cur_id}\t{len(inliers)}\t\t{np.sum(remain_points)}')
            cur_id += 1
    
        return plane_id, line_id

def stop_compress(covered_points, k):
    return np.sum(covered_points < k) == 0

def get_weights(plane_id, line_id):
    assert np.sum((plane_id >= 0) * (line_id >= 0)) == 0
    N = len(plane_id)
    plane_num = int(np.max(plane_id)+1)
    line_num = int(np.max(line_id)+1)
    
    sigma_plane = [np.sum(plane_id == i) / N for i in range(plane_num)]
    sigma_line = [np.sum(line_id == i) / N for i in range(line_num)]
    sigma_rest = np.sum((plane_id == -1) * (line_id == -1)) / N
    if VERBOSE: print(f'portion of non plane / line points: {sigma_rest * 100:.02f}%')
    
    weight = []
    for i in range(N):
        if plane_id[i] != -1:
            weight.append(sigma_plane[plane_id[i]])
        elif line_id[i] != -1:
            weight.append(sigma_line[line_id[i]])
        else:
            weight.append(sigma_rest)
            
    return np.array(weight) / np.sum(weight)

def compress_model(points3d, image_id, k=100, method='vanilla', min_plane_points=500, min_line_points=50, dxyz=None):
    # Initialize
    compressed_model = np.zeros(len(points3d))
    cameras = np.max(np.concatenate(image_id)) + 1
    covered_points = np.zeros(cameras)
    
    # Detect planes and lines
    plane_id, line_id = structure_detection(points3d, min_plane_points=min_plane_points, min_line_points=min_line_points, method=method, dxyz=dxyz)
    
    # Assign weights
    weights = get_weights(plane_id, line_id)
    
    # Get visibility
    visibility = np.zeros((cameras, len(points3d)), dtype=int)
    for i in range(len(image_id)):
        visible_cameras = image_id[i]
        for c in visible_cameras:
            visibility[c, i] = 1
    for c in range(cameras):
        # print(f'visible points of camera {c}: {np.sum(visibility[c])}')
        if np.sum(visibility[c]) < k:
            covered_points[c] += k - np.sum(visibility[c])

    if VERBOSE: sp = 0
    while not stop_compress(covered_points, k):
        weights /= np.sum(weights)
        score = np.sum(weights * visibility, axis=0)
        score[compressed_model > 0] = 0
        max_point = np.argmax(score)
        compressed_model[max_point] = 1
        for c in image_id[max_point]:
            covered_points[c] += 1
        pid, lid = plane_id[max_point], line_id[max_point]
        if pid >= 0:
            # selected plane gets lower weight
            weights[plane_id == pid] /= 2
        elif lid >= 0:
            # selected line gets lower weight
            weights[line_id == lid] /= 2
        else:
            # non-plane/line get lower weight
            weights[(plane_id < 0) * (line_id < 0)] /= 2
        
        remaining_valid_points = (weights > 0) * (compressed_model == 0)
        if np.sum(remaining_valid_points) == 0:
            if VERBOSE: print('not enough remaining points')
            break
        remaining_valid_index = np.where(remaining_valid_points)[0]
        for i in remaining_valid_index:
            if stop_compress(covered_points[visibility[:, i].reshape(-1) > 0], k):
                weights[i] = 0
        
        if VERBOSE: 
            sp += 1
            isplane = plane_id[max_point]
            isline = line_id[max_point]
            if isplane >= 0:
                source_msg = f'plane {isplane}'
            elif isline >= 0:
                source_msg = f'line {isline}'
            else:
                source_msg = 'non plane/line'
            print(f'{sp}: selected point from {source_msg} ({max_point}), camera done: {np.sum(covered_points >= k)}/{cameras}, remaining valid points: {np.sum(remaining_valid_points)}/{len(points3d)}', end='\033[K\r')
    if VERBOSE: print(f'compression done, points after compression: {sp}/{len(points3d)}\033[K')
    return points3d[compressed_model > 0], (compressed_model > 0)

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path of input model')
    parser.add_argument('-m', '--method', type=str, required=True, help='method to compress')
    parser.add_argument('-s', '--save', type=str, required=True, help='path to save compressed model')
    parser.add_argument('-k', '--K', type=int, required=True, help='minimum number of points to preserve in each view')
    parser.add_argument('-mpp', '--min-plane-points', type=int, required=True, help='minimum number of points to determine a plane')
    parser.add_argument('-mlp', '--min-line-points', type=int, required=True, help='minimum number of points to determine a line')
    parser.add_argument('-dxyz', '--dxyz', type=int, nargs=3, default=[3, 3, 3], help='side length of the blocks to partition the point cloud if method is LOCAL')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity, prints messages if set true')    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # get arguments
    args = _get_args()
    
    # set verbosity
    VERBOSE = args.verbose
    
    # load model
    with open(args.input, 'rb') as f:
        model = pickle.load(f)

    # id, xyz, rgb, image_id, descriptor 
    model = np.array(model, dtype=object)
    ID = model[:, 0].astype(int)
    xyz = np.vstack(model[:, 1])
    rgb = np.vstack(model[:, 2])
    image_id = model[:, 3]
    descriptor = np.vstack(model[:, 4])
    
    # compress model
    compressed_points, compressed_idx = compress_model(xyz, image_id, k=args.K, method=args.method, 
                                                       min_plane_points=args.min_plane_points, min_line_points=args.min_line_points, 
                                                       dxyz=args.dxyz)
    compressed_model = model[compressed_idx]
    
    with open(args.save, 'wb') as f:
        pickle.dump(compressed_model, f)
        
    '''
    example usage:
    python compressModel.py -v -i '../SFM2DB/model.pkl' -m vanilla -s compressed_model_100.pkl -k 100 -mpp 500 -mlp 100
    python compressModel.py -v -i '../SFM2DB/model.pkl' -m pca -s compressed_model_pca_100.pkl -k 100 -mpp 500 -mlp 100
    python compressModel.py -v -i '../SFM2DB/model.pkl' -m local -s compressed_model_local_100.pkl -k 100 -mpp 100 -mlp 20 -dxyz 5 5 5
    '''