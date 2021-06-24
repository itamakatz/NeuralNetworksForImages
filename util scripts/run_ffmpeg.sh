#!/bin/bash
echo arg1: $1
echo arg2: $2
echo starting!
ffmpeg -framerate 600 -i $1 -c:v libx264 -profile:v high -crf 1 -s 64x64 -pix_fmt yuv420p $2
# ffmpeg -framerate 60 -i $1 -c:v libx264 -profile:v high -crf 1 -s 64x64 -pix_fmt yuv420p $2
echo Finished