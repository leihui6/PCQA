# Dataset

Here we explain how we collect and process the point clouds

## 1. Synthesis Data

You can find these point clouds in the ./synthesis directory, which includes cube, curved surface, and plane data. To explicitly remesh and standardize the size and aspect ratio of the mesh, we use the following:

``` python
def normalization(data, minValue = None):
    if minValue == None:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    elif minValue == 0:
        _range = np.max(data) - minValue
        return data / _range
        
def calculateDistanceTwoPoints(p1, p2):
    return np.sqrt(np.sum((p1-p2) * (p1-p2)))

for filename in tqdm(fileList):
    edgeList = dict()
    print (f'loading {filename}.STL')
    mesh = o3d.io.read_triangle_mesh(f'{filename}.STL')
    triangleList = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    print (f'info: vertices length: {vertices.shape}')
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
    print (f'calculation {filename}.STL done!')
    distancePerPoints, totalDistance = [], []
    print (f'edgeList: {len(edgeList)}')
    for key in edgeList:
        p = key
        others = edgeList[key]
        ds = [calculateDistanceTwoPoints(vertices[p], vertices[op]) for op in others]
        distancePerPoints.append(np.mean(ds))
        totalDistance += ds
    print (f'filename: {filename},\n\
    mean distance for each point: {np.around(np.mean(distancePerPoints), 3)},\n\
    std:{np.around(np.std(distancePerPoints), 3)},\n\
    cv: {np.around(100 * np.std(distancePerPoints)/np.mean(distancePerPoints), 3)}%')
```

## 2. Scanning Repository

You can check the `scanning_repository.zip` in `./scanning_repository` and the sparse point clouds are generated by

``` python
fileList = [
    './blade',
    './bunny',
    './dragon',
    './hand',
    './happy',
    './horse',
    './angle',
    './armadillo'
]

ratioList = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]

np.random.seed(42)
for filename in tqdm(fileList):
    points = np.loadtxt(f'{filename}.xyz')[:,0:3]
    print (f'points({filename}.xyz) shape: {points.shape}')
    cktree = cKDTree(points, leafsize = 16)
    
    for keepRatio in ratioList:
        totalN = points.shape[0]
        keepIndices = np.random.choice(range(totalN), int(totalN * keepRatio), replace=False)
        print (f'keep points: {len(points)} ({np.around(keepRatio, 3) * 100}%)')
        remainPoints = points[keepIndices]
        print (f'remain points: {remainPoints.shape}')
        np.savetxt(f'./sampled/{filename}_sampled_{np.around(keepRatio, 2)}.xyz', remainPoints)
    print ('')
```

## 3. Public Dataset

The dataset we used for evaluation are [SJTU-PCQA](https://vision.nju.edu.cn/28/fd/c29466a469245/page.htm), [LS-PCQA](https://smt.sjtu.edu.cn/database/large-scale-point-cloud-quality-assessment-dataset-ls-pcqa/) and [WPC](https://github.com/qdushl/Waterloo-Point-Cloud-Database). We recommend visiting their website for direct access. Here, we only tested the downsampled point clouds.

## 4. Density Calculation

Below is a code snippet demonstrating how we calculate the density of point clouds. The density calculation methods used for these point clouds are:

1. **Using fixed radius to calculate the density**

    ```python
    fileList = [
        'plane-isometric',
    ]
    print (f'here is filelist to be inspected (number: {len(fileList)}):\n{fileList}')
    pathFileList = [
        './plane0-isometric_pathPoints_in'
    ]
    stdCvList = []
    for i, filename in enumerate(fileList):
        points = np.loadtxt(f'{filename}.xyz')[:,0:3].astype(np.float64)
        pathPoints = np.loadtxt(f'{filename}.xyz')[:,0:3].astype(np.float64)
        print (f'points({filename}.xyz) format: {points.shape}')
        specificNum = 5000 # it is the number of points we want to calculate in order to save time
        if specificNum >= points.shape[0]:
            sampledIndices = np.linspace(0, points.shape[0] - 1,
                                        int(points.shape[0])).astype(np.int32)
        else:
            sampledIndices = np.linspace(0, points.shape[0] - 1, 
                                        int(0.5 * points.shape[0])).astype(np.int32)
        print (f'calculated points length: {len(sampledIndices)}')
        tree = cKDTree(points)
        meanDistances = []
        for pi in sampledIndices:
            dd, ii = tree.query(points[pi], k = 30)
            meanDistances.append(np.mean(dd))
        meanDis = np.mean(meanDistances)
        searchR = meanDis
        densityList = []
        for pi in sampledIndices:
            indices = tree.query_ball_point(points[pi], r = searchR)
            # density in a ball
            density = len(indices) / ((4/3) * np.pi * np.power(searchR, 3))
            densityList.append(density)

        densityList = normalization(densityList, 0)
        stdValue, cvValue = np.std(densityList), np.std(densityList)/np.mean(densityList)
        print (f'mean distance: {np.mean(densityList)}')
        print (f'{stdValue}\n{cvValue}')
        stdCvList.append([stdValue, cvValue])
    ```

2. **Using an adaptive radius to calculate the density**

    ``` python
    fileList = [
        'plane-isometric',
    ]

    print (f'here is filelist to be inspected (number: {len(fileList)}):\n{fileList}')

    stdCvList = []
    for filename in tqdm(fileList):
        points = np.loadtxt(f'{filename}.xyz')[:,0:3].astype(np.float64)
        print (f'points({filename}) format: {points.shape}')
        specificNum = 5000 # it is the number of points we want to calculate in order to save time
        if specificNum >= points.shape[0]:
            sampledIndices = np.linspace(0, points.shape[0] - 1,
                                        int(points.shape[0])).astype(np.int32)
        else:
            sampledIndices = np.linspace(0, points.shape[0] - 1, 
                                        int(0.5 * points.shape[0])).astype(np.int32)
        print (f'calculated points length: {len(sampledIndices)}')
        tree = cKDTree(points)
        meanDistances, densityList = [], []
        for pi in sampledIndices:
            dd, ii = tree.query(points[pi], k = 30)
            # tmpMeanValue = np.mean(dd) # try another way to calculate the dynamic radius
            tmpMeanValue = np.max(dd)
            indices = tree.query_ball_point(points[pi], r = tmpMeanValue)
            density = len(indices) / ((4/3) * np.pi * np.power(tmpMeanValue, 3))
            densityList.append(density)
        densityList = normalization(densityList, 0)
        stdValue, meanValue = np.std(densityList), np.mean(densityList)
        print (f'{stdValue}\n{stdValue/meanValue}')
        stdCvList.append([stdValue, stdValue/meanValue])
    ```