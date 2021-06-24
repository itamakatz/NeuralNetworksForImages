# ffmpeg -framerate 60 -i %04d.png -c:v libx264 -profile:v high -crf 1 -s 64x64 -pix_fmt yuv420p output_60fps.mp4

import os
import cv2
import shutil
import argparse
from os import listdir
from natsort import natsorted
from os.path import isfile, join

def process_path(relative_path, force_rename):

  dir_path = join(os.path.dirname(os.path.realpath(__file__)), relative_path)

  onlyfiles = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and join(dir_path, f).endswith(".png") and "iter" not in f and "batch_count" in f]
  if(len(onlyfiles) == 0): 
      print("No files found..")
      return
  onlyfiles = natsorted(onlyfiles)

  if(force_rename and os.path.exists(join(dir_path, "Renamed"))):
    shutil.rmtree(join(dir_path, "Renamed"))

  if not os.path.exists(join(dir_path, "Renamed")):
    os.makedirs(join(dir_path, "Renamed"))

    for i, file in enumerate(onlyfiles):
      img = cv2.imread(file)
      if(img.shape[0] == 64 and img.shape[1] == 64):
        res = img
      else:
        res = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_NEAREST)
      cv2.imwrite(join(dir_path, "Renamed", f"{i:06d}.png"), res)

  images_path = join(dir_path, "Renamed", "%06d.png")
  max_ = max(relative_path.split("/"), key=len)
  video_name = join(os.path.dirname(os.path.realpath(__file__)), "Videos", max_ + "_60fps.mp4")

  os.system(f'./run_ffmpeg.sh "{images_path}" "{video_name}"')

if __name__ == '__main__':

  os.chdir(os.path.dirname(os.path.realpath(__file__)))

  parser = argparse.ArgumentParser(description='create video')

  parser.add_argument('-p', '--path', type=str, default="", metavar=('path'),
              help=f'Sets relative path to the directory. If the path is empty (this is the default) then all sub directories will be processed. (default: "")')
  parser.add_argument('--force_rename', action='store_true', default=False,
            help=f'Deletes the old renamed images if they exist and create them again (default: {False})')

  args = parser.parse_args()
  
  print("Starting")

  if not os.path.exists(join(os.path.dirname(os.path.realpath(__file__)), "Videos")):
    os.makedirs(join(os.path.dirname(os.path.realpath(__file__)), "Videos"))

  if(args.path == ""):
    for path in [x[0] for x in os.walk(os.path.dirname(os.path.realpath(__file__)))]:
      process_path(path, args.force_rename)
  else:
    process_path(args.path, args.force_rename)