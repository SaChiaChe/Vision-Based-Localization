import os
import cv2
import argparse
import numpy as np

# Parameters used for cv2.goodFeaturesToTrack (Shi-Tomasi Features)
feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# CONSTANTS
fMATCHING_DIFF = 1  # Minimum difference in the KLT point correspondence

lk_params = dict(winSize=(21, 21),  # Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker)
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path ot input video')
    parser.add_argument('-c', '--camera_parameters', default='camera_parameters.npy', type=str, help='npy file of camera parameters')
    parser.add_argument('-s', '--save', required=True, type=str, help='path to save extracted frames')
    parser.add_argument('-p', '--min-pixel-diff', default=10, type=float, help='minimum difference to extract frames')
    parser.add_argument('-n', '--num-frame', default=0, type=int, help='extract frame with fixed offset')
    args = parser.parse_args()
    return args



def KLT_featureTracking(image_ref, image_cur, px_ref):
    """Feature tracking using the Kanade-Lucas-Tomasi tracker.
    A backtracking check is done to ensure good features. The backtracking features method
    consist of tracking a set of features, f-1, onto a new frame, which will produce the corresponding features, f-2,
    on the new frame. Once this is done we take the f-2 features, and look for their
    corresponding features, f-1', on the last frame. When we obtain the f-1' features we look for the
    absolute difference between f-1 and f-1', abs(f-1 and f-1'). If the absolute difference is less than a certain
    threshold(in this case 1) then we consider them good features."""

    # Feature Correspondence with Backtracking Check
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params)

    d = abs(px_ref - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
    good = d < fMATCHING_DIFF  # Verify which features produced good results by the difference being less
                               # than the fMATCHING_DIFF threshold.
    # Error Management
    if len(d) == 0:
        print('Error: No matches where made.')
    elif list(good).count(
            True) <= 5:  # If less than 5 good points, it uses the features obtain without the backtracking check
        print('Warning: No match was good. Returns the list without good point correspondence.')
        return kp1, kp2, np.inf

    # Create new lists with the good features
    n_kp1, n_kp2 = [], []
    for i, good_flag in enumerate(good):
        if good_flag:
            n_kp1.append(kp1[i])
            n_kp2.append(kp2[i])

    # Format the features into float32 numpy arrays
    n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

    # Verify if the point correspondence points are in the same
    # pixel coordinates. If true the car is stopped (theoretically)
    d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

    # The mean of the differences is used to determine the amount
    # of distance between the pixels
    diff_mean = np.mean(d)

    return n_kp1, n_kp2, diff_mean

def betterMatches(F, points1, points2):
    """ Minimize the geometric error between corresponding image coordinates.
    For more information look into OpenCV's docs for the cv2.correctMatches function."""

    # Reshaping for cv2.correctMatches
    points1 = np.reshape(points1, (1, points1.shape[0], 2))
    points2 = np.reshape(points2, (1, points2.shape[0], 2))

    newPoints1, newPoints2 = cv2.correctMatches(F, points1, points2)

    return newPoints1[0], newPoints2[0]

class featureTracker:
    def __init__(self, prev_frame, min_diff, detector='SHI-TOMASI'):
        self.prev_frame = prev_frame
        self.min_diff = min_diff
        self.detector = detector
        self.px_ref = self.detectNewFeatures(self.prev_frame)

    def detectNewFeatures(self, cur_img):
        """Detects new features in the current frame.
        Uses the Feature Detector selected."""
        if self.detector == 'SHI-TOMASI':
            feature_pts = cv2.goodFeaturesToTrack(cur_img, **feature_params)
            feature_pts = np.array([x for x in feature_pts], dtype=np.float32).reshape((-1, 2))
        else:
            feature_pts = self.detector.detect(cur_img, None)
            feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)

        return feature_pts

    def track(self, cur_frame):
        px_ref, px_cur, px_diff = KLT_featureTracking(self.prev_frame, cur_frame, self.px_ref)
        if not self.skip_frame(px_diff):
            self.prev_frame = cur_frame
            self.px_ref = self.detectNewFeatures(self.prev_frame)
            return True

        return False

    def skip_frame(self, px_diff):
        return px_diff < self.min_diff

def extract_frames(video_path, save_path, min_pix_diff, K=None, dist=None):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Failed to open video ({video_path})')
        exit()

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    if dist is not None:
        gray_frame = cv2.undistort(gray_frame, K, dist)
    tracker = featureTracker(gray_frame, min_pix_diff)
    for i in range(1, num_frames):
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if dist is not None:
            gray_frame = cv2.undistort(gray_frame, K, dist)
        if tracker.track(gray_frame):
            cv2.imwrite(os.path.join(save_path, f'frame_{i:04d}.png'), frame)
            
def extract_frames_fixed(video_path, save_path, N=15):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Failed to open video ({video_path})')
        exit()

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frames):
        ret, frame = cap.read()
        if i % N == 0:
            cv2.imwrite(os.path.join(save_path, f'frame_{i:04d}.jpg'), frame)


if __name__ == "__main__":
    args = get_args()
    
    N = args.num_frame
    if N == 0:
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        K = camera_params['K']
        dist = camera_params['dist']
        extract_frames(args.input, args.save, args.min_pixel_diff, K, dist)
    else:
        extract_frames_fixed(args.input, args.save, N)
