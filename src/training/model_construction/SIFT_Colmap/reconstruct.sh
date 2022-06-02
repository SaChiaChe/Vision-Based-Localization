#!/bin/bash
PROJ_PATH='./'
IMG='./images'
DB='./database.db'
SPARSE='./sparse'
DENSE='./dense'
MININLIER='1000'

if [ -z "$1" ]
then
	echo "Use following argument:"
	echo "build - automatic reconstructor"
	echo "feaext - feature extraction output database"
	echo "exhmat - exhaustive matching"
	echo "mapper - construct sparse"
	echo "undist - image undistorter construct dense"
	echo "stereo - patch_match_stereo"
	echo "fusion - stereo fusion construct structure"
	echo "pomesh - poisson mesher"
	echo "demesh - delaunay_mesher"
	echo "clean - remove build file but keep image dir"
fi
if [ "$1" = "build" ]
then
	colmap feature_extractor \
		--database_path $DB \
		--image_path $IMG \
		--ImageReader.camera_model OPENCV \
		--ImageReader.single_camera 1
	colmap exhaustive_matcher \
		--database_path $DB
	mkdir $SPARSE
	if [ -z "$2" ]
	then
		echo "min inliers 1000"
	else
		MININLIER="$2"
		echo "min inliers $2"
	fi 
	colmap mapper \
		--database_path $DB \
		--image_path $IMG \
		--output_path $SPARSE \
		--Mapper.init_min_num_inliers $MININLIER
	mkdir $DENSE
	colmap image_undistorter \
		--image_path $IMG \
		--input_path $SPARSE/0 \
		--output_path $DENSE \
		--output_type COLMAP \
		--max_image_size 2000
	colmap patch_match_stereo \
		--workspace_path $DENSE \
		--workspace_format COLMAP \
		--PatchMatchStereo.geom_consistency true
	colmap stereo_fusion \
		--workspace_path $DENSE \
		--workspace_format COLMAP \
		--input_type geometric \
		--output_path $DENSE/fused.ply \
		#--output_type TXT
	mkdir model
	mv $DENSE/fused.ply $DENSE/sparse/* ./database.db ./model
elif [ "$1" = "feaext" ]
then
	colmap feature_extractor \
		--database_path $DB \
		--image_path $IMG \
		--ImageReader.camera_model OPENCV \
		--ImageReader.single_camera 1
elif [ "$1" = "exhmat" ]
then
	colmap exhaustive_matcher \
		--database_path $DB
elif [ "$1" = "mapper" ]
then
	mkdir $SPARSE
	if [ -z "$2" ]
	then
		echo "min inliers 1000"
	else
		MININLIER="$2"
		echo "min inliers $2"
	fi 
	colmap mapper \
		--database_path $DB \
		--image_path $IMG \
		--output_path $SPARSE \
		--Mapper.init_min_num_inliers $MININLIER
elif [ "$1" = "undist" ]
then
	mkdir $DENSE
	colmap image_undistorter \
		--image_path $IMG \
		--input_path $SPARSE/0 \
		--output_path $DENSE \
		--output_type COLMAP \
		--max_image_size 2000
elif [ "$1" = "stereo" ]
then
	colmap patch_match_stereo \
		--workspace_path $DENSE \
		--workspace_format COLMAP \
		--PatchMatchStereo.geom_consistency true
elif [ "$1" = "fusion" ]
then
	colmap stereo_fusion \
		--workspace_path $DENSE \
		--workspace_format COLMAP \
		--input_type geometric \
		--output_path $DENSE/fused.ply \
		#--output_type TXT
elif [ "$1" = "pomesh" ]
then
	colmap poisson_mesher \
		--input_path $DENSE/fused.ply \
		--output_path $DENSE/meshed-poisson.ply
elif [ "$1" = "demesh" ]
then
	colmap delaunay_mesher \
		--input_path $DENSE \
		--output_path $DENSE/meshed-delaunay.ply
elif [ "$1" = "supercolmap" ]
then
	mkdir $DENSE
	colmap image_undistorter \
		--image_path $IMG \
		--input_path $SPARSE/0 \
		--output_path $DENSE \
		--output_type COLMAP \
		--max_image_size 2000
	colmap patch_match_stereo \
		--workspace_path $DENSE \
		--workspace_format COLMAP \
		--PatchMatchStereo.geom_consistency true
	colmap stereo_fusion \
		--workspace_path $DENSE \
		--workspace_format COLMAP \
		--input_type geometric \
		--output_path $DENSE/fused.ply
elif [ "$1" = "clean" ]
then
	rm -r $DB $SPARSE $DENSE
fi
