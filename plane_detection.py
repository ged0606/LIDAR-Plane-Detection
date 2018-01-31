from numba import jit
import numpy as np
import pandas as pd
import random
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from sklearn.decomposition import PCA
import sys
import yaml

def minimum_rectangle(points_3d):
    plane_center = np.average(points_3d, axis = 0)
    points_3d -= plane_center
    
    # Get basis vectors for plane
    covariance = np.cov(points_3d, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    v1 = eigvecs[:, 1]
    v2 = eigvecs[:, 2]
    
    basis_change = np.vstack((v1, v2))
    
    projection = points_3d.dot(basis_change.T)
    try:
        hull = ConvexHull(projection)
        hull_vertices = projection[hull.vertices]

        edges = np.zeros((len(hull_vertices)-1, 2))
        edges = hull_vertices[1:] - hull_vertices[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        
        pi2 = np.pi / 2.0
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)
        
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-pi2),
            np.cos(angles+pi2),
            np.cos(angles)
        ]).T
        rotations = rotations.reshape((-1, 2, 2))
        
        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_vertices.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)
        
        rval_3d = rval.dot(basis_change)
        
        return rval_3d + plane_center
    except QhullError:
        return []

def sort_lidar_file_and_shape(lidar_file_name, yaml_file_name, width=2088, height=64):
    df = pd.read_csv(lidar_file_name,
                     names = ['x', 'y', 'z', 'intensity', 'ring', 'rotation', 'revolution'])
    
    yaml_file = yaml.load(open(yaml_file_name))
    
    calibration = yaml_file['lasers']
    
    sorted_lasers = sorted(calibration, key=lambda x: x['vert_correction'], reverse=True)

    for i in range(0, 64):
        df.loc[df['ring'] == i, 'rotation'] = (df.loc[df['ring'] == i, 'rotation'] 
                                               + sorted_lasers[i]['rot_correction'] * 18000 / np.pi)
    
    df.loc[df['rotation'] > 36000, 'rotation'] -= 36000
    df.loc[df['rotation'] < 0, 'rotation'] += 36000
    
    img = np.zeros((height, width, 3))
    for i in range(height):
        img[i] = df.loc[df['ring'] == i].sort_values(['rotation']).as_matrix()[:, :3]
    
    return img

@jit
def calculate_normals(lidar_table, depths, neighbor_y_radius=9, neighbor_x_radius=9,
                      width=2088, height=64):
    normals = np.zeros(lidar_table.shape)
    for i in range(neighbor_y_radius, height-neighbor_y_radius - 1):
        for j in range(0, width):        
            neighbor_y_start = i - neighbor_y_radius
            neighbor_y_end = i + neighbor_y_radius + 1
            neighbor_x = np.arange(j - neighbor_x_radius, j + neighbor_x_radius + 1)
            
            neighbors = lidar_table[neighbor_y_start:neighbor_y_end]
            neighbors = np.take(neighbors, neighbor_x, axis = 1, mode='wrap')
            neighbors = neighbors.reshape(((neighbor_y_radius * 2 + 1) * (neighbor_x_radius * 2 + 1), 3))
            
            neighbor_depths = depths[neighbor_y_start:neighbor_y_end]
            neighbor_depths = np.take(neighbor_depths, neighbor_x, axis=1, mode='wrap')
            neighbor_depths = neighbor_depths.ravel()
            
            valid_neighbors = neighbors[np.abs(neighbor_depths - depths[i, j]) < 0.2]
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
            neighbor_x = np.arange(j - neighbor_x_radius, j + neighbor_x_radius + 1)
            
            neighbors = normals[neighbor_y_start:neighbor_y_end]
            neighbors = np.take(neighbors, neighbor_x, axis = 1, mode='wrap')
            neighbors = neighbors.reshape(((neighbor_y_radius * 2 + 1) * (neighbor_x_radius * 2 + 1), 3))
            
            neighbor_depths = depths[neighbor_y_start:neighbor_y_end]
            neighbor_depths = np.take(neighbor_depths, neighbor_x, axis=1, mode='wrap')
            neighbor_depths = neighbor_depths.ravel()
            
            valid_neighbors = neighbors[np.abs(neighbor_depths - depths[i, j]) < 0.2]
            normals[i, j] = np.average(valid_neighbors, axis = 0)
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
        self.normal = (len(self.points) * self.normal + len(cluster.points) * cluster.normal) \
                      / (len(self.points) + len(cluster.points))
        self.normal /= np.linalg.norm(self.normal)
        self.points += cluster.points
        # Do PCA only when the cluster has at least self.pca_min points
        #if len(self.points) >= self.pca_min:
        #    covariance = np.cov(np.array(self.points), rowvar=False)
        #    eigvals, eigvecs = np.linalg.eigh(covariance)
        #    self.normal = eigvecs[:, 0]
        #    self.normal /= np.linalg.norm(self.normal)
        #    if np.dot(self.normal, np.array([0, 1, 0]) - 
        #          self.points[0]) <= 0:
        #        self.normal = -self.normal
        #else:
        #    self.normal = (self.normal * (len(self.points) - len(cluster.points))  
        #                 + cluster.normal * len(cluster.points)) / len(self.points)

def cluster_points_by_normals(lidar_table, normal_img, depths, 
                              neighbor_y_radius=5,
                              width=2088, height=64):

    cluster_assignments = [ [Cluster(normal_img[i, j], i * normal_img.shape[1] + j, lidar_table[i, j]) 
                             for j in range(normal_img.shape[1])]
                           for i in range(normal_img.shape[0]) ]
    id_to_cluster = []
    for row in cluster_assignments:
        for cluster in row:
            id_to_cluster.append(cluster)
    
    cluster_id_to_coords = [[(i // normal_img.shape[1], i % normal_img.shape[1])]
                            for i in range(normal_img.shape[0] * normal_img.shape[1])]
    
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
                                if diff < 0.2 and angle < .06:
                                    merges[merge] = angle
                            elif diff < 0.2:
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
            if len(cluster.points) >= 3 and cluster.normal[2] < 0.7:
                hull = minimum_rectangle(np.array(cluster.points))
                if len(hull) > 0:
                    side1_length = np.linalg.norm(hull[0] - hull[1])
                    side2_length = np.linalg.norm(hull[1] - hull[2])
                    longside = max(side1_length, side2_length)
                    shortside = min(side1_length, side2_length)
                    if longside >= 0.8 and longside <= 1.0 and shortside >= 0.5 and shortside <= 0.7:
                        boards.append(hull)
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
        if boards is not None:
            for board_rectangle in boards:
                for i in range(len(board_rectangle)):
                    point1 = board_rectangle[i]
                    point2 = board_rectangle[(i + 1) % len(board_rectangle)]
                    f3.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(point1[0], point1[1], point1[2],
                                                                           point2[0], point2[1], point2[2],
                                                                           1, 1, 1))

        else:
            print("Calibration board not detected.")

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

