#!/bin/bash

echo ">[BUILD]<START>< Downloading KITTI datasets..."
TARGET_DIR=data

rm $TARGET_DIR/2011_09_26_drive_0002_extract.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_extract.zip -P $TARGET_DIR
unzip -o $TARGET_DIR/2011_09_26_drive_0002_extract.zip -d $TARGET_DIR
rm $TARGET_DIR/2011_09_26_drive_0002_extract.zip

rm $TARGET_DIR/2011_09_26_drive_0002_sync.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip -P $TARGET_DIR
unzip -o $TARGET_DIR/2011_09_26_drive_0002_sync.zip -d $TARGET_DIR
rm $TARGET_DIR/2011_09_26_drive_0002_sync.zip

rm $TARGET_DIR/2011_09_26_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip -P $TARGET_DIR
unzip -o $TARGET_DIR/2011_09_26_calib.zip -d $TARGET_DIR
rm $TARGET_DIR/2011_09_26_calib.zip

rm $TARGET_DIR/2011_09_26_drive_0002_tracklets.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_tracklets.zip -P $TARGET_DIR
unzip -o $TARGET_DIR/2011_09_26_drive_0002_tracklets.zip -d $TARGET_DIR
rm $TARGET_DIR/2011_09_26_drive_0002_tracklets.zip

echo ">[BUILD]<DONE>< Downloading KITTI datasets..."

