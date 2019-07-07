import numpy as np
import colorsys
import math
from enum import Enum
import open3d as o3d
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
import csg_c

import hdbscan


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


def normalize_pc(pc):
    x_max, y_max, z_max, _, _, _, _, _, _, _, _ = pc.max(axis=0)
    x_min, y_min, z_min, _, _, _, _, _, _, _, _ = pc.min(axis=0)

    print(str(x_max) + " " + str(y_max) + " " + str(z_max))  # + " " + str(label_max))
    print(str(x_min) + " " + str(y_min) + " " + str(z_min))  # + " " + str(label_min))
    # print(str(label_max) + " " + str(label_min))

    f = np.max([abs(x_max - x_min), abs(y_max - y_min), abs(z_max - z_min)])

    pc[:, 0:3] /= f
    pc[:, 3:6] /= np.linalg.norm(pc[:, 3:6], ord=2, axis=1, keepdims=True)

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


def write_clusters(path, labels, data):
    file = open(path, "w")

    file.write(str(len(np.unique(labels))) + "\n")

    clusters = {}
    i = 0
    for l in labels:

        clusters.setdefault(str(l), []).append(data[i,:9])
        i += 1


    for key, points in clusters.items():

        manifold_type = ManifoldType.NONE
        if points[0][6] > 0.5:
            manifold_type = ManifoldType.CYLINDER
        elif points[0][7] > 0.5:
            manifold_type = ManifoldType.SPHERE
        elif points[0][8] > 0.5:
            manifold_type = ManifoldType.PLANE

        file.write(str(manifold_type.value) + "\n")
        file.write(str(len(points)) + " 6\n")
        for p in points:
            file.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]) + " " + str(p[5]) + "\n")

if __name__ == "__main__":

    _, manifolds = read_manifolds("C:/Projekte/csg_playground_build/ransac_res.txt")

    data = []

    for m in manifolds:
        print("Type: " + str(m[0]))

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

            if phi > 1.0 or phi < 0.0:
                print(phi)

            p.append(theta)
            p.append(phi)

            # t = #float(m[0].value)
            # p.append(t)
            # print(p)
            data.append(p)

    print("Number of manifolds: " + str(len(manifolds)))

    data = np.array(data)

    normalize_pc(data)
    print("PC normalized.")


    # csg_c.ransac(np.array([[1.0,2.0,3.0,4.0,5.0,6.0],[7.0,8.0,9.0,10.0,11.0,12.0]], dtype=np.float32))#data[:,:6])
    # csg_c.ransac(np.array(data[:,:6])) #we need to do the copy here, standard type is double
    # print(data[:2,:6])

    # print(data[:,[0,1,2,6]])

    def similarity(x, y):

        # return np.linalg.norm(x - y)

        dist = np.linalg.norm(x[:3] - y[:3])

        label_dist = float(x[6] != y[6])

        angle_dist = 0.0

        # print(str(dist) + " " + str(label_dist))

        return dist + angle_dist + label_dist


    dbscan = DBSCAN(eps=0.1)  # , metric=similarity)

    optics = OPTICS(min_samples=200,
                    xi=0.5,
                    min_cluster_size=2.0)

    cluster_algo = dbscan
    db = cluster_algo.fit(data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]])
    labels = db.labels_

    write_clusters("test.txt", labels, data)

    # clusterer = clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    # labels = clusterer.fit_predict(data[:,[0,1,2,3,4,5,6,7,8]])

    colors = np.array(3, dtype=float)

    N = len(set(labels))
    c = [colorsys.hsv_to_rgb(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    colors = np.zeros(shape=(len(labels), 3))
    i = 0
    for label in labels:
        colors[i, :] = c[label]
        i += 1

    # colors /= float(len(set(labels)))
    # print(colors)

    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True

    # unique_labels = set(labels)

    # for unique_label in unique_labels:

    # class_member_mask = (labels == unique_label)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Clusters: " + str(n_clusters) + " Noise: " + str(n_noise))

    data2 = farthest_points(data[:, :3], 1000)
    print("Data: " + str(len(data2)))

    # print(data[:, :3])

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(data[:, :3])
    pcd.colors = o3d.Vector3dVector(colors)

    # pcd = read_point_cloud("model_1.xyzn")

    # print(pcd)
    # print(np.asarray(pcd.points))
    o3d.draw_geometries([pcd])
