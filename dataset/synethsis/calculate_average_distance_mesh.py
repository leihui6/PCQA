import numpy as np
import open3d as o3d
from tqdm import tqdm

if __name__ == "__main__":

    fileList = [
        #     './cube2-isometric',
        #     './drillBit-isometric',
        #     './hypoidGear-isometric',
        "./hypoidGear-isometric"
    ]

    def calculateDistanceTwoPoints(p1, p2):
        return np.sqrt(np.sum((p1 - p2) * (p1 - p2)))

    np.set_printoptions(precision=6)

    for filename in tqdm(fileList):
        edgeList = dict()
        print(f"loading {filename}.STL")
        mesh = o3d.io.read_triangle_mesh(f"{filename}.STL")
        triangleList = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        print(f"info: vertices length: {vertices.shape}")
        for triangle in triangleList:
            p, p1, p2 = triangle
            if p not in edgeList:
                edgeList[p] = []
            if p1 not in edgeList:
                edgeList[p1] = []
            if p2 not in edgeList:
                edgeList[p2] = []

            # p to p1 p2
            if p1 not in edgeList[p]:
                edgeList[p].append(p1)
            if p2 not in edgeList[p]:
                edgeList[p].append(p2)

            # p1 to p p2
            if p2 not in edgeList[p1]:
                edgeList[p1].append(p2)
            if p not in edgeList[p1]:
                edgeList[p1].append(p)

            # p2 to p p1
            if p1 not in edgeList[p2]:
                edgeList[p2].append(p1)
            if p not in edgeList[p2]:
                edgeList[p2].append(p)
        print(f"calculation {filename}.STL done!")
        distancePerPoints, totalDistance = [], []
        print(f"edgeList: {len(edgeList)}")
        for key in edgeList:
            p = key
            others = edgeList[key]
            ds = [calculateDistanceTwoPoints(vertices[p], vertices[op]) for op in others]
            distancePerPoints.append(np.mean(ds))
            totalDistance += ds
        print(
            f"filename: {filename},\n\
        mean distance for each point: {np.around(np.mean(distancePerPoints), 3)},\n\
        std:{np.around(np.std(distancePerPoints), 3)},\n\
        cv: {np.around(100 * np.std(distancePerPoints)/np.mean(distancePerPoints), 3)}%"
        )
