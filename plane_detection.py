import matplotlib.path as mplPath
from numba import jit
import numpy as np
import pandas as pd
import random
import scipy.optimize
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from sklearn.decomposition import PCA
import sys
import yaml


def sort_lidar_file_and_shape(lidar_file_name, yaml_file_name,
                              width=2088, height=64):
    df = pd.read_csv(lidar_file_name,
                     names=['x', 'y', 'z', 'intensity', 'ring', 'rotation', 'revolution'])

    yaml_file = yaml.load(open(yaml_file_name))

    calibration = yaml_file['lasers']

    sorted_lasers = sorted(calibration, key=lambda x: x['vert_correction'], reverse=True)

    img = np.zeros((height, width, 3))
    for i in range(height):
        img[i] = df.loc[df['ring'] == i] \
                 .sort_values(['rotation']) \
                 .as_matrix()[:, :3]

    return img.astype(np.float32)


@jit
def calculate_normals(lidar_table, depths,
                      neighbor_y_radius=9, neighbor_x_radius=9,
                      width=2088, height=64):
    normals = np.zeros(lidar_table.shape)
    for i in range(neighbor_y_radius, height-neighbor_y_radius - 1):
        for j in range(0, width):        
            neighbor_y_start = i - neighbor_y_radius
            neighbor_y_end = i + neighbor_y_radius + 1
            neighbor_x = np.arange(j - neighbor_x_radius, j + neighbor_x_radius + 1)

            neighbors = lidar_table[neighbor_y_start:neighbor_y_end]
            neighbors = np.take(neighbors, neighbor_x, axis=1, mode='wrap')
            neighbors = neighbors.reshape(((neighbor_y_radius * 2 + 1) * 
                                          (neighbor_x_radius * 2 + 1), 3))

            neighbor_depths = depths[neighbor_y_start:neighbor_y_end]
            neighbor_depths = np.take(neighbor_depths, neighbor_x, axis=1,
                                      mode='wrap')
            neighbor_depths = neighbor_depths.ravel()

            valid_neighbors = \
                neighbors[np.abs(neighbor_depths - depths[i, j]) < 0.08]

            if valid_neighbors.shape[0] >= 3:
                covariance = np.cov(valid_neighbors, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(covariance)
                normals[i, j] = eigvecs[:, 0]
                normals[i, j] /= np.linalg.norm(normals[i, j])
                if np.dot(normals[i, j], np.array([0, 1, 0]) -
                      lidar_table[i, j]) <= 0:
                    normals[i, j] = -normals[i, j]

    # Average normals
    for i in range(neighbor_y_radius, height-neighbor_y_radius - 1):
        for j in range(0, width):
            neighbor_y_start = i - neighbor_y_radius
            neighbor_y_end = i + neighbor_y_radius + 1
            neighbor_x = np.arange(j - neighbor_x_radius,
                                   j + neighbor_x_radius + 1)

            neighbors = normals[neighbor_y_start:neighbor_y_end]
            neighbors = np.take(neighbors, neighbor_x, axis=1, mode='wrap')
            neighbors = neighbors.reshape(((neighbor_y_radius * 2 + 1) *
                                          (neighbor_x_radius * 2 + 1), 3))

            neighbor_depths = depths[neighbor_y_start:neighbor_y_end]
            neighbor_depths = np.take(neighbor_depths, neighbor_x, axis=1,
                                      mode='wrap')
            neighbor_depths = neighbor_depths.ravel()

            valid_neighbors = \
                neighbors[np.abs(neighbor_depths - depths[i, j]) < 0.1]
            normals[i, j] = np.average(valid_neighbors, axis=0)
    return normals


class Cluster:
    def __init__(self, normal, num, point, pca_min=6):
        self.normal = normal
        self.id = num
        self.points = [point]
        self.evecs = None
        self.pca_min = pca_min
        
    def add_normal(self, normal):
        self.normal = (self.normal * self.count + normal) / (self.count + 1)
        self.count += 1
        
    def remove_normal(self, normal):
        self.normal = (self.normal * self.count - normal) / (self.count - 1)

    def merge_with(self, cluster):
        self.normal = (len(self.points) * self.normal + 
                       len(cluster.points) * cluster.normal) / \
                      (len(self.points) + len(cluster.points))
        self.normal /= np.linalg.norm(self.normal)
        self.points += cluster.points


def cluster_points_by_normals(lidar_table, normal_img, depths,
                              neighbor_y_radius=5,
                              width=2088, height=64):
    cluster_assignments = [[Cluster(normal_img[i, j], i * normal_img.shape[1] + j, lidar_table[i, j]) 
                             for j in range(normal_img.shape[1])]
                           for i in range(normal_img.shape[0])]
    id_to_cluster = []
    for row in cluster_assignments:
        for cluster in row:
            id_to_cluster.append(cluster)

    cluster_id_to_coords = [[(i // normal_img.shape[1],
                              i % normal_img.shape[1])]
                            for i in range(normal_img.shape[0] *
                                           normal_img.shape[1])]

    dont_check = np.zeros((normal_img.shape[0], normal_img.shape[1]))

    while True:
        merges = dict()

        # Get potential merges for each element by looking at pixel neighbors 
        for x in range(0, normal_img.shape[0] - 1):
            for y in range(normal_img.shape[1]):
                if dont_check[x, y]:
                    continue

                dont_check_current = 1
                cluster = cluster_assignments[x][y]
                normal = cluster.normal
                new_cluster = None

                for dx in range(0, 4):
                    for dy in range(0, 4):
                        new_x = (x + dx) % normal_img.shape[0]

                        new_y = (y + dy) % normal_img.shape[1]

                        temp_cluster = cluster_assignments[new_x][new_y]
                        if temp_cluster.id != cluster.id:
                            dont_check_current = 0

                            merge = (min(temp_cluster.id, cluster.id), 
                                     max(temp_cluster.id, cluster.id)) 

                            angle = np.arccos(np.dot(normal, temp_cluster.normal) / 
                                              np.linalg.norm(normal) / 
                                              np.linalg.norm(temp_cluster.normal))

                            diff = np.abs(depths[new_x, new_y] - depths[x, y])

                            if merge not in merges:
                                if diff < 0.07 and angle < .06:
                                    merges[merge] = angle
                            elif diff < 0.07:
                                merges[merge] = min(angle, merges[merge])

                dont_check[x, y] = dont_check_current

        print("Potential Merges: {}".format(len(merges)))
        if len(merges) == 0:
            break

        # Find best merges for each cluster and merge, starting with lowest cluster ids
        potential_merges = sorted(merges.keys(), key = lambda x: x[0])
        current_cluster = potential_merges[0][0]
        min_merge_candidate = potential_merges[0][1]
        min_merge_angle = np.pi
        for i in range(1, len(potential_merges) + 1):
            if i == len(potential_merges) or potential_merges[i][0] != current_cluster:
                coords = cluster_id_to_coords[current_cluster]
                cluster_id_to_coords[current_cluster] = []
                new_cluster = id_to_cluster[min_merge_candidate]
                for coord in coords:
                    cluster_assignments[coord[0]][coord[1]] = new_cluster
                new_cluster.merge_with(id_to_cluster[current_cluster])
                cluster_id_to_coords[min_merge_candidate] += coords
                min_merge_angle = np.pi
                if i < len(potential_merges) - 1:
                    min_merge_candidate = potential_merges[i + 1][1]
                    current_cluster = potential_merges[i + 1][0]
            else:
                if merges[potential_merges[i]] < min_merge_angle:
                    min_merge_candidate = potential_merges[i][1]
                    min_merge_angle = merges[potential_merges[i]]        

    valid_cluster_assignments = cluster_assignments[neighbor_y_radius:height-neighbor_y_radius - 1]
    valid_lidars = lidar_table[neighbor_y_radius:height-neighbor_y_radius - 1]

    # Add points to clusters
    valid_cluster_ids = set()
    valid_clusters = []
    for i in range(len(valid_cluster_assignments)):
        for j in range(len(valid_cluster_assignments[i])):
            cluster = valid_cluster_assignments[i][j]
            if cluster.id not in valid_cluster_ids:
                valid_cluster_ids.add(cluster.id)
                valid_clusters.append(cluster)
    return valid_clusters

bottom_left= np.array([-.254, -.381])
top_left = np.array([-.254, .381])
top_right = np.array([.254, .381])
bottom_right = np.array([.254, -.381])
board = np.vstack((bottom_left, top_left, top_right, bottom_right))

def find_boards(clusters):
    def cost_function_for_points(projection):
        def func(a):
            x = a[0]
            y = a[1]
            theta = a[2]
            rot = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])
            rectangle = np.matmul(board, rot.T) + a[:2]
            
            distances = np.zeros((projection.shape[0], 
                                  rectangle.shape[0]))
            for j in range(rectangle.shape[0]):
                p1 = rectangle[j] 
                p2 = rectangle[(j + 1) % rectangle.shape[0]]
                distances[:, j] = (p2[1] - p1[1]) * projection[:, 0] - (p2[0] - p1[0]) * projection[:, 1] + p2[0] * p1[1] - p2[1] * p1[0]
                distances[:, j] = np.abs(distances[:, j]) / np.linalg.norm(p2 - p1)
            return np.sum(np.min(distances, axis=1))
        return func
    
    boards = []
    for cluster in clusters:
        if len(cluster.points) < 3:
            continue

        points = np.array(cluster.points)
        plane_center = np.average(points, axis=0)
        points -= plane_center

        # Get basis vectors for plane
        covariance = np.cov(points, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        v1 = eigvecs[:, 1]
        v2 = eigvecs[:, 2]

        basis_change = np.vstack((v1, v2))

        projection = points.dot(basis_change.T)
        
        try:
            hull = ConvexHull(projection)
            hull_vertices = projection[hull.vertices]

            cost_func = cost_function_for_points(hull_vertices)
            sol = scipy.optimize.least_squares(cost_func, (0, 0, 0))
            cost = sol.cost

            theta = sol.x[2]
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rect = np.matmul(board, rot.T) + sol.x[:2]

            AB = rect[1] - rect[0]
            BC = rect[2] - rect[1]
            AM = projection - rect[0]
            BM = projection - rect[1]
            AB_AM = np.matmul(AM, AB)
            AB_AB = AB.dot(AB)
            BC_BM = BM.dot(BC)
            BC_BC = BC.dot(BC)
            
            valid = np.logical_and(np.logical_and(AB_AM >= 0, AB_AM <= AB_AB), np.logical_and(BC_BM >= 0, BC_BM <= BC_BC))

            cost += projection.shape[0] - 2 * np.sum(valid)

            rect3d = rect.dot(basis_change) + plane_center
            boards.append((cost, rect3d))
        except QhullError:
            pass

    boards.sort(key = lambda x: x[0])
    return [b[1] for b in boards[:5]]

def detect_planes(lidar_file_name, calibration_file_name, width=2088, height=64,
                  points_file_name="cluster_points.csv", board_file_name="cluster_lines.csv",
                  plane_ply_file_name="detected_planes.ply", normal_ply_file_name="normals.ply"):

    lidar_table = sort_lidar_file_and_shape(lidar_file_name, calibration_file_name)
    
    print("Calculate Depths")
    depths = np.linalg.norm(lidar_table, axis=-1)

    print("Calculate Normals")
    normal_img = calculate_normals(lidar_table, depths)

    print("Clustering")
    clusters = cluster_points_by_normals(lidar_table, normal_img, depths)

    cluster_colors = [(random.random(), random.random(), random.random()) 
                      for i in range(len(clusters))]
  
    boards = []
    minimum_dist = float('inf')
    
    red = np.array([1, 0, 0])
    yellow = np.array([1, 1, 0])
    green = np.array([0, 1, 0])
    blue = np.array([0, 0, 1])

    with open(points_file_name, 'w') as f1, open(plane_ply_file_name, 'w') as f2, \
         open(board_file_name, "w") as f3, open(normal_ply_file_name, 'w') as f4:
        num_points = 0
        for i in range(len(clusters)):
            cluster = clusters[i]
            color = cluster_colors[i]
            num_points += len(cluster.points)
            for point in cluster.points:
                #dist = np.linalg.norm(point)
                #if dist < 4.0:
                #    dist /= 4.0
                #    color = (1.0 - dist) * red + dist * yellow
                #elif dist < 8.0:
                #    dist -= 4.0
                #    dist /= 4.0
                #    color = (1.0 - dist) * yellow + dist * green
                #else:
                #    dist -= 8.0
                #    dist /= 4.0
                #    if dist < 1:
                #        color = (1.0 - dist) * green + dist * blue
                #    else:
                #        color = blue
                    
                f1.write("{}, {}, {}, {}, {}, {}\n".format(point[0], point[1], point[2], 
                                                           color[0], color[1], color[2]))
        # Plane ply file
        f2.write("ply\n")
        f2.write("format ascii 1.0\n")
        f2.write("element vertex {}\n".format(num_points))
        f2.write("property float32 x\n")
        f2.write("property float32 y\n")
        f2.write("property float32 z\n")
        f2.write("property float32 nx\n")
        f2.write("property float32 ny\n")
        f2.write("property float32 nz\n")
        f2.write("end_header\n")

        for i in range(len(clusters)):
            cluster = clusters[i]
            normal = cluster.normal
            for point in cluster.points:
                f2.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], 
                                                           normal[0], normal[1], normal[2]))
        boards = find_boards(clusters)
        for board_rectangle in boards:
            for i in range(len(board_rectangle)):
                point1 = board_rectangle[i]
                point2 = board_rectangle[(i + 1) % len(board_rectangle)]
                f3.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(point1[0], point1[1], point1[2],
                                                                       point2[0], point2[1], point2[2],
                                                                       1, 1, 1))

        # Normal ply file
        f4.write("ply\n")
        f4.write("format ascii 1.0\n")
        f4.write("element vertex {}\n".format(num_points))
        f4.write("property float32 x\n")
        f4.write("property float32 y\n")
        f4.write("property float32 z\n")
        f4.write("property float32 nx\n")
        f4.write("property float32 ny\n")
        f4.write("property float32 nz\n")
        f4.write("end_header\n")
        for i in range(lidar_table.shape[0]):
            for j in range(lidar_table.shape[1]):
                point = lidar_table[i][j]
                normal = normal_img[i][j]
                f4.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], 
                                                           normal[0], normal[1], normal[2]))


if __name__ == "__main__":
    detect_planes(sys.argv[1], sys.argv[2])
