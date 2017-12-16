import plane_detection
import os

calibration_file = '64HDL_S2.yaml'

directory = '../../../Downloads/2017-11-04-11-04-17_0/'
files = os.listdir(directory)

save_dir = 'detection/'
ind = 0

point_file_name_template = save_dir + "points/{}.csv"
board_file_name_template = save_dir + "boards/{}.csv"
ply_file_name_template = save_dir + "ply/{}.ply"
for fname in sorted(files):
  print(fname)
  plane_detection.detect_planes(directory+fname, calibration_file, width=2088, height=64,
                points_file_name=point_file_name_template.format(ind), board_file_name=board_file_name_template.format(ind),
                ply_file_name=ply_file_name_template.format(ind))
  ind+=1
