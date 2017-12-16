import sys

normal_ply_file_name = sys.argv[1]

with open(normal_ply_file_name, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex {}\n".format(len(normal_points)))
    f.write("property float32 x\n")
    f.write("property float32 y\n")
    f.write("property float32 z\n")
    f.write("property float32 nx\n")
    f.write("property float32 ny\n")
    f.write("property float32 nz\n")
    f.write("end_header\n")
    for i in range(len(normal_points)):
        normal = normal_colors[i]
        f.write("{} {} {} {} {} {}\n".format(normal_points[i][0], normal_points[i][1], normal_points[i][2],
                                             normal[0], normal[1], normal[2]))
