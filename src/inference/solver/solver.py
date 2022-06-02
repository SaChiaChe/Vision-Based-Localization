from enum import Enum
import cv2 as cv
import numpy as np

class PoseSolver(object):
    class SolveMethod(Enum):
        PNP = "pnp"
        EPNP = "epnp"
        P3P = "p3p"
        AP3P = "ap3p"
        def __str__(self) -> str:
            return self.value
        
    method2flag = {
        SolveMethod.PNP: cv.SOLVEPNP_ITERATIVE,
        SolveMethod.EPNP: cv.SOLVEPNP_EPNP,
        SolveMethod.P3P: cv.SOLVEPNP_P3P,
        SolveMethod.AP3P: cv.SOLVEPNP_AP3P
    }

    def __init__(self, method: SolveMethod, intrinsic, distcoeff, use_ransac):
        self.flag = self.method2flag[method]
        self.intrinsic = intrinsic
        self.distcoeff = distcoeff
        self.use_ransac = use_ransac

    def __call__(self, object_points, image_points, **kwargs):
        inliers = None
        if self.use_ransac:
            retval, rvec, tvec, inliers = cv.solvePnPRansac(object_points, image_points, self.intrinsic, self.distcoeff, flags=self.flag, reprojectionError=8.0, **kwargs)
            if not retval:
                print('opencvPnPRansac Failed')
                return 
            object_points = object_points[inliers.flatten()]
            image_points = image_points[inliers.flatten()]
        else:
            retval, rvec, tvec = cv.solvePnP(object_points, image_points, self.intrinsic, self.distcoeff, flags=self.flag)

        err = self.reprojection_error(object_points, image_points, rvec, tvec)
        if err > 25:
            raise ValueError(f'reprojection error {err} with {len(inliers.flatten())} inliers')
        #     print(retval, inliers, err)
        return rvec, tvec, err, inliers

    def reprojection_error(self, object_points, image_points, rvec, tvec):
        reprojected_points, _ = cv.projectPoints(object_points, rvec, tvec, self.intrinsic, self.distcoeff)        
        reprojected_points = reprojected_points.reshape(-1, 2)
        # rmat = cv.Rodrigues(rvec)[0]
        # extrinsic = np.concatenate([rmat, tvec], -1)
        # projection_mat = self.intrinsic @ extrinsic
        # reprojected_points = projection_mat @ np.concatenate((object_points, np.ones((object_points.shape[0], 1))), axis=1).T
        # reprojected_lambda = 1 / np.tile(reprojected_points[2, :], (3, 1))
        # reprojected_points = reprojected_points * reprojected_lambda
        err = np.mean(np.linalg.norm(reprojected_points - image_points, axis=1))

        return err