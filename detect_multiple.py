import plane_detection
import os

calibration_file = '64HDL_S2.yaml'

directory = '../test_lidar/'
files = os.listdir(directory)

save_dir = 'detection/'
ind = 0

point_file_name_template = save_dir + "points/{}.csv"
board_file_name_template = save_dir + "boards/{}.csv"
plane_ply_file_name_template = save_dir + "ply/{}-planes.ply"
normal_ply_file_name_template = save_dir + "ply/{}-normal.ply"
for fname in sorted(files):
  if fname != ".DS_Store":
    print(fname)
    plane_detection.detect_planes(directory+fname, calibration_file, width=2088, height=64,
                  points_file_name=point_file_name_template.format(ind), 
                  board_file_name=board_file_name_template.format(ind),
                  plane_ply_file_name=plane_ply_file_name_template.format(ind),
                  normal_ply_file_name=normal_ply_file_name_template.format(ind))
    ind+=1
