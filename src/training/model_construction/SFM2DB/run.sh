#!/bin/bash
if [ -z "$1" ]
then
	echo "Use following argument:"
	echo "SIFT - For model constructed by SIFT feature extraction"
	echo "Superpoint - For model constructed by SuperPoint feature extraction"
fi

if [ "$1" = "SIFT" ]
then
	python3 ./CreateDB.py --input_model_path ./model/ --images_path ./images --DB_path ./model/database.db
fi

if [ "$1" = "Superpoint" ]
then
	python3 ./CreateDB.py --method Superpoint --input_model_path ./model/ --images_path ./images --DB_path ./model/database.db
fi

