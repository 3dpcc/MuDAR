# MuDAR: Multi-dimensional Context Mining for LiDAR Point Cloud Compression

## ✨ Introduction

LiDAR point cloud compression (PCC) is essential for networked machinery, e.g., autonomous vehicles and robotics, to cache and exchange high-precision positioning data for perception. Existing methods, including volumetric- and octree-based approaches, are limited by insufficient retrieval of correlated neighboring points for effective context modeling. This paper introduces MuDAR, a multi-dimensional context mining-based LiDAR PCC, which jointly exploits correlations by gathering proximate neighbors from multi-dimensional spaces, including an octree-structured 1D array, a LiDAR scanning-related 2D map, and a Cartesian coordinates indexed 3D spatial tensor, for more accurate and efficient context modeling. Extensive experiments validate the effectiveness of MuDAR. On average, it attains 43\% and 47\% compression gains over the standardized G-PCC in static and dynamic coding modes, respectively, across four widely used LiDAR datasets.

## ⚙️ Environment Install 

### Download pytorch

```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install lightning
```
### Download some components
open3d: ```pip install open3d```

tqdm: ```pip install tqdm```

pandas: ```pip install pandas```

## 😀 Usage

### Training

### Testing


## Authors
These files are provided by Hangzhou Normal University [3DPCC](https://github.com/3dpcc) and Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). Please contact us (zhujiahao23@stu.hznu.edu.cn) if you have any questions.


