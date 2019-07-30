import numpy as np
import colorsys
import math
from enum import Enum
import open3d as o3d
from sklearn.cluster import DBSCAN


class ManifoldType(Enum):
    NONE = 0
    CYLINDER = 1
    SPHERE = 2
    PLANE = 3


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def farthest_points(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(float(int(i[:2], 16)) / 255.0, float(int(i[2:4], 16)) / 255.0, float(int(i[4:], 16)) / 255.0) for i in
            colors]


def get_manifold_type(p_type):
    if 'cylinder' in p_type.lower():
        return ManifoldType.CYLINDER
    elif 'sphere' in p_type.lower():
        return ManifoldType.SPHERE
    elif 'plane' in p_type.lower():
        return ManifoldType.PLANE

    return ManifoldType.NONE


def read_pc(file):
    header = file.readline()
    num_points = int(header.split()[0])
    pc = []

    for i in range(num_points):
        pc.append(list(float(s) for s in file.readline().split()))

    return pc


def normalize_pc(pc, label_weight):
    x_max, y_max, z_max, _, _, _, _, _, _, _, _ = pc.max(axis=0)
    x_min, y_min, z_min, _, _, _, _, _, _, _, _ = pc.min(axis=0)

    print(str(x_max) + " " + str(y_max) + " " + str(z_max))  # + " " + str(label_max))
    print(str(x_min) + " " + str(y_min) + " " + str(z_min))  # + " " + str(label_min))
    # print(str(label_max) + " " + str(label_min))

    f = np.max([abs(x_max - x_min), abs(y_max - y_min), abs(z_max - z_min)])

    pc[:, 0:3] /= f

    pc[:, 3:6] /= (np.linalg.norm(pc[:, 3:6], ord=2, axis=1, keepdims=True))

    #pc[:, 6:9] *= 1024

    #print( pc[:10, 6:9])

    return pc


def read_manifolds(path):
    file = open(path)

    pc = read_pc(file)

    num_manifolds = int(file.readline())
    manifolds = []

    for i in range(num_manifolds):
        manifold_type = get_manifold_type(file.readline().split()[0])
        manifold_pc = read_pc(file)
        manifolds.append((manifold_type, manifold_pc))

    return pc, manifolds


def write_clusters(path, data, subcluster_types):

    total_clusters = []

    clusters = extract_clusters(data, False)

    for cluster in clusters:

        manifold_type = cluster[0]
        points = cluster[1]

        if manifold_type in subcluster_types:
            sub_clusters = extract_clusters(points, True)
            for sub_cluster in sub_clusters:
                total_clusters.append(sub_cluster)
        else:
            total_clusters.append(cluster)

    file = open(path, "w")
    file.write(str(len(total_clusters)) + "\n")
    for cluster in total_clusters:
        write_cluster(file,cluster)

    return total_clusters


def extract_clusters(data, sub_cluster_mode):

    dbscan = DBSCAN(eps=0.1)

    db = None
    if sub_cluster_mode is True:
        db = dbscan.fit(data[:, [ 0, 1, 2,3, 4, 5]])
    else:
        db = dbscan.fit(data[:, [ 0, 1, 2,6, 7, 8]])

    labels = db.labels_

    clusters = {}
    i = 0
    for l in labels:
        clusters.setdefault(str(l), []).append(data[i, :9])
        i += 1

    res_clusters = []
    for key, points in clusters.items():

        manifold_type = ManifoldType.NONE
        if points[0][6] > 0.0001:
            manifold_type = ManifoldType.CYLINDER
        elif points[0][7] > 0.0001:
            manifold_type = ManifoldType.SPHERE
        elif points[0][8] > 0.0001:
            manifold_type = ManifoldType.PLANE


        res_clusters.append((manifold_type, np.array(points)))

    return res_clusters

def write_cluster(file, cluster):
    file.write(str(cluster[0].value) + "\n")
    file.write(str(len(cluster[1])) + " 6\n")
    for p in cluster[1]:
        file.write(
            str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]) + " " + str(p[5]) + "\n")

if __name__ == "__main__":
    label_weight = 32.0
    _, manifolds = read_manifolds("C:/Projekte/csg_playground_build/ransac_res.txt")

    data = []

    for m in manifolds:
        #print("Type: " + str(m[0]))

        for p in m[1]:

            if m[0] is ManifoldType.CYLINDER:
                p.append(1.0)
                p.append(0.0)
                p.append(0.0)
            elif m[0] is ManifoldType.SPHERE:
                p.append(0.0)
                p.append(1.0)
                p.append(0.0)
            elif m[0] is ManifoldType.PLANE:
                p.append(0.0)
                p.append(0.0)
                p.append(1.0)
            else:
                p.append(0.0)
                p.append(0.0)
                p.append(0.0)

            theta = math.acos(p[5]) / math.pi
            phi = (math.atan2(p[4], p[3]) + math.pi) / (2.0 * math.pi)

            p.append(theta)
            p.append(phi)

            data.append(p)

    print("Number of manifolds: " + str(len(manifolds)))

    data = np.array(data)

    normalize_pc(data, label_weight)

    clusters = write_clusters("test.txt", data, [ManifoldType.SPHERE, ManifoldType.PLANE])

    print("Number of clusters: " + str(len(clusters)))
    for cluster in clusters:
        print("Cluster " + str(cluster[0]))

    colors = np.array(3, dtype=float)
    N = len(clusters)
    c = [colorsys.hsv_to_rgb(x * 1.0 / N, 0.5, 0.5) for x in range(N)]

    num_points = 0
    for cluster in clusters:
        num_points += len(cluster[1])
    colors = np.zeros(shape=(num_points, 3))
    points = np.zeros(shape=(num_points, 9))
    point_idx = 0
    cluster_idx = 0
    for cluster in clusters:
        for point in cluster[1]:
            colors[point_idx, :] = c[cluster_idx]
            points[point_idx] = point
            point_idx += 1
        cluster_idx += 1

    #data2 = farthest_points(data[:, :3], 1000)
    #print("Data: " + str(len(data2)))

    # print(data[:, :3])

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(points[:, :3])
    pcd.colors = o3d.Vector3dVector(colors)

    # pcd = read_point_cloud("model_1.xyzn")

    # print(pcd)
    # print(np.asarray(pcd.points))
    o3d.draw_geometries([pcd])
