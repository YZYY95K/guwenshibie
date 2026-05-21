#!/bin/bash
set -e

BUILD_DIR=/root/project/docker_build
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR/src $BUILD_DIR/models/det $BUILD_DIR/models/rec

cp /root/project/run_inference.py $BUILD_DIR/src/
cp /root/project/Dockerfile $BUILD_DIR/
cp /root/project/requirements.txt $BUILD_DIR/
cp /root/project/run.sh $BUILD_DIR/

cp /root/project/models/detection/yolov8s_char_det_v2/weights/best.pt $BUILD_DIR/models/det/
cp /root/project/models/recognizer/exp040_swin_small_merged_hybrid_2stage_ema/best.pth $BUILD_DIR/models/rec/
cp /root/project/data/processed/crops_merged/char_to_id.json $BUILD_DIR/models/rec/

echo "Build directory ready:"
du -sh $BUILD_DIR/*
du -sh $BUILD_DIR

cd $BUILD_DIR
docker build -t guwen-ocr:latest .

echo "Docker image built successfully!"
docker images | grep guwen-ocr

echo ""
echo "To save as tar for submission:"
echo " docker save guwen-ocr:latest | gzip > /root/project/guwen-ocr-exp040.tar.gz"
BUILDEOF
chmod +x /root/project/build_docker.sh
echo build_docker.sh updated