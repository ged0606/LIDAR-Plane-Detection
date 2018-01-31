import sys

csv_file_name = sys.argv[1]
ply_file_name = sys.argv[2]

with open(csv_file_name, 'r') as csv_file:
    lines = csv_file.readlines()
    with open(ply_file_name, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(lines)))
        f.write("property float32 x\n")
        f.write("property float32 y\n")
        f.write("property float32 z\n")
        f.write("end_header\n")
        for line in lines:
            vals = line.split(',')
            f.write("{} {} {}\n".format(vals[0], vals[1], vals[2]))
        f.close()
