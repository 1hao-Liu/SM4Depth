## Models

| name | description | link |  
| :---: | :---: | :---: |
| SM4Depth | used in paper | [baidu](https://pan.baidu.com/s/1navbYdCY5qZhjlURHARIiQ?pwd=5cky)  |
| Swin-Base | SimMIM pretarined  | [baidu](https://pan.baidu.com/s/1NQ7AeD7X8PxSIkJMs0I1PA?pwd=vra4) |


## Validation Sets

| name | scene | capture | size | link |  
| :---: | :---: | :---: | :---: | :---: |
| NYUD | indoor | RGB-D | 654 | [baidu](https://pan.baidu.com/s/12e2xRw_xI_Fz_Ix5LbJ4_w?pwd=r2f6) |
| KITTI | outdoor | LiDAR | 652 | [baidu](https://pan.baidu.com/s/1YUQ1c2zjI_idgt4wwhug7w?pwd=3a2u) |

## Test Sets

| name | scene | capture | size | link |  
| :---: | :---: | :---: | :---: | :---: |
| SUN RGB-D | indoor | RGB-D | 4,395 | [baidu](https://pan.baidu.com/s/1jqTc0ZAHGtZUPNQs6GMN2Q?pwd=hb7n) |
| iBims-1 | indoor | LiDAR | 100 | [baidu](https://pan.baidu.com/s/14o2PmUXwlILB6jK2NR_J-w?pwd=y489) |
| ETH3D | both | LiDAR | 454 | [baidu](https://pan.baidu.com/s/1i8w8JQOjiOc5Z9RS1EmPOw?pwd=qgfr) |
| DIODE | both | LiDAR | 771 | [baidu](https://pan.baidu.com/s/1F5e5RdICPmVtQD09V6MCpg?pwd=yrhc) |
| nuScenes-val | outdoor | LiDAR | 1,138 | [baidu](https://pan.baidu.com/s/1npO3v_GviQUEn4KiAQXZMg?pwd=uvdj) |
| DDAD | outdoor | LiDAR | 3,950 | [baidu](https://pan.baidu.com/s/1idvRM--yrA69lppqr14sFA?pwd=a350) |
| **BUPT Depth** | both | Stereo*# | 14,932 | [baidu](https://pan.baidu.com/s/1_ld6iz3gcyQ1CzmmuICA5Q?pwd=g6gd) |

> \* : regenerate depth by CreStereo
> 
> \# : remove sky regions by ViT-Adapter

## Intrinsics

| name | intrinsics: [fx, fy, cx, cy] |
| :---: | :---: |
| [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | [518.857901, 519.469611, 284.582449, 208.736166] |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) | [707.0493, 707.0493, 604.0814, 180.5066] |
| [SUN RGBD](https://rgbd.cs.princeton.edu/) ( kv2d ) | [529.5, 529.5, 365.0, 265.0] |
| SUN RGBD ( kv2a ) | [529.5, 529.5, 365.0, 265.0] |
| SUN RGBD ( kv1N=NYUD ) | [518.857901, 519.469611, 284.582449, 208.736166] |
| SUN RGBD ( kv1b ) | [520.532, 520.7444, 277.9258, 215.115] |
| SUN RGBD ( xtis ) | [570.342205, 570.342205, 310, 225] |
| SUN RGBD ( xtix ) | [570.342224, 570.342224, 291, 231] |
| SUN RGBD ( relg ) | [693.74469, 693.74469, 360.431915, 264.75] |
| SUN RGBD ( resa ) | [693.74469, 693.74469, 360.431915, 264.75] |
|[iBims-1](https://www.asg.ed.tum.de/lmf/ibims1/)| [550.39, 548.55, 355.44, 240.26] |
|[ETH3D](https://www.eth3d.net/datasets) | [3415.0, 3415.0, 3100.0, 2070.0] (approximately) |
|[DIODE](https://diode-dataset.org/)| [886.81, 927.06, 512, 384] |
|[DDAD](https://github.com/TRI-ML/DDAD)| [2181.53025, 2181.60344, 928.02188, 615.95678] |
|[nuScenes](https://www.nuscenes.org/nuscenes)| [1266.4172, 1266.4172, 800, 450] |