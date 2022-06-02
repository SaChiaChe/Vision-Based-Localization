 #!/bin/bash

MODEL=$1 # "../../model/campus1.pkl" "../../model/campus_sp.pkl"
COMPRESS_MODEL_NAME=$2 # "../../model/campus1" "../../model/campus_sp"

for K in 20 50 100 200 300
do
    echo -ne "compress model vanilla K=${K}\n"
    python compressModel.py -v -i ${MODEL} -m vanilla -s ${COMPRESS_MODEL_NAME}_${K}.pkl -k ${K} -mpp 500 -mlp 100
    
    echo -ne "compress model pca K=${K}\n"
    python compressModel.py -v -i ${MODEL} -m pca -s ${COMPRESS_MODEL_NAME}_pca_${K}.pkl -k ${K} -mpp 500 -mlp 100
    
    echo -ne "compress model local K=${K}\n"
    python compressModel.py -v -i ${MODEL} -m local -s ${COMPRESS_MODEL_NAME}_local_${K}.pkl -k $K -mpp 100 -mlp 20 -dxyz 3 3 3
done