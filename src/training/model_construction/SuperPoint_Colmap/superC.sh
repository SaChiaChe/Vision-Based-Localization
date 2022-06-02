#!/bin/bash
python super_colmap.py \
      --projpath ./projdir/ \
      --cameraModel RADIAL \
      --images_path images \
      --single_camera
cp ./reconstruct.sh ./projdir/
( cd ./projdir/ ; ./reconstruct.sh supercolmap )
mkdir model
cp ./projdir/database.db ./projdir/dense/sparse/* ./projdir/dense/fused.ply ./model
