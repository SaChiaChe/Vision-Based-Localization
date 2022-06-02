import os

WIDTHS = [1920, 960, 640]
HEIGHTS = [1080, 540, 360]
CAMERA_PARAMS = ['camera_parameter/camera_parameters_mike.npy', 'camera_parameter/camera_parameters_downsample_2.npy', 'camera_parameter/camera_parameters_downsample_3.npy']
K = ['_20', '_50', '_100', '_200'] # ['_20', '_50', '_100', '_200']
SOLVE_METHODS = ['epnp'] # ['epnp', 'pnp', 'p3p', 'ap3p']
MATCH_METHODS = ['brute_force'] # ['brute_force', 'kd_tree']
COMPRESS_METHODS = ['_pca'] # ['', '_pca', '_local']
sift_model_name = '../model/campus1' # 'campus1'
superpoint_model_name = '../model/campus_sp' # 'campus_sp'
input_path = '../data/video/campus3' # campus3

if __name__ == "__main__":
    for i in range(len(HEIGHTS)): # Different image size / camera parameters
        for s in SOLVE_METHODS: # Different pose estimation methods
            for m in MATCH_METHODS: # Different matching methods
                # No compress
                print(f"input: campus3 | model: {os.path.split(sift_model_name)[-1]} | feature: sift | match: {m} | solver: {s} | size: {WIDTHS[i]}x{HEIGHTS[i]}")
                cmd = f'python inference.py --input {input_path} \
                                            --camera_parameters {CAMERA_PARAMS[i]} \
                                            --useSparse --sparse_model_file {sift_model_name}.pkl \
                                            --feature_type sift \
                                            --match_method {m} --use_ransac\
                                            --calculation_method {s} \
                                            --init_rvec 0.01953391 -0.159423 0.03049893 \
                                            --init_tvec 1.95370437 0.13115526 -1.2827208 \
                                            --view_scale -30 \
                                            --resize {WIDTHS[i]} {HEIGHTS[i]} '
                try:
                    os.system(cmd)
                except:
                    pass

                print(f"input: campus3 | model: {os.path.split(superpoint_model_name)[-1]} | feature: superpoint | match: {m} | solver: {s} | size: {WIDTHS[i]}x{HEIGHTS[i]}")
                cmd = f'python inference.py --input {input_path} \
                                            --camera_parameters {CAMERA_PARAMS[i]} \
                                            --useSparse --sparse_model_file {superpoint_model_name}.pkl \
                                            --feature_type superpoint \
                                            --match_method {m} --use_ransac\
                                            --calculation_method {s} \
                                            --init_rvec 0.05056287 -0.0922757 0.0504959 \
                                            --init_tvec 1.86439232 0.1080795 -1.17792502 \
                                            --view_scale -30 \
                                            --resize {WIDTHS[i]} {HEIGHTS[i]} '
                try:
                    os.system(cmd)
                except:
                    pass

                for c in COMPRESS_METHODS: # Different compression methods
                    for k in K: # Different compression K
                        # Compression
                        print(f"input: campus3 | model: {os.path.split(sift_model_name)[-1]}{c}{k} | feature: sift | match: {m} | solver: {s} | size: {WIDTHS[i]}x{HEIGHTS[i]}")
                        cmd = f'python inference.py --input {input_path} \
                                                    --camera_parameters {CAMERA_PARAMS[i]} \
                                                    --useSparse --sparse_model_file {sift_model_name}{c}{k}.pkl \
                                                    --feature_type sift \
                                                    --match_method {m} --use_ransac\
                                                    --calculation_method {s} \
                                                    --init_rvec 0.01953391 -0.159423 0.03049893 \
                                                    --init_tvec 1.95370437 0.13115526 -1.2827208 \
                                                    --view_scale -20 \
                                                    --resize {WIDTHS[i]} {HEIGHTS[i]} '
                        try:
                            os.system(cmd)
                        except:
                            pass

                        print(f"input: campus3 | model: {os.path.split(superpoint_model_name)[-1]}{c}{k} | feature: superpoint | match: {m} | solver: {s} | size: {WIDTHS[i]}x{HEIGHTS[i]}")
                        cmd = f'python inference.py --input {input_path} \
                                                    --camera_parameters {CAMERA_PARAMS[i]} \
                                                    --useSparse --sparse_model_file {superpoint_model_name}{c}{k}.pkl \
                                                    --feature_type superpoint \
                                                    --match_method {m} --use_ransac\
                                                    --calculation_method {s} \
                                                    --init_rvec 0.05056287 -0.0922757 0.0504959 \
                                                    --init_tvec 1.86439232 0.1080795 -1.17792502 \
                                                    --view_scale -20 \
                                                    --resize {WIDTHS[i]} {HEIGHTS[i]} '
                        try:
                            os.system(cmd)
                        except:
                            pass
