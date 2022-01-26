# SISNet
Semantic Scene Completion via Integrating Instances and Scene in-the-Loop (CVPR 2021)

In this repository, we provide SISNet model implementation (with Pytorch) as well as data preparation, training and evaluation scripts on NYU Depth V2, NYUCAD and SUNCG_RGBD.

![image](https://github.com/yjcaimeow/SISNet/blob/main/figs/nyu_vis.png)

## Code is under construction.

## Getting Started
### Set up
Clone the repository:

    git clone https://github.com/yjcaimeow/SISNet
### Installation
The code is tested with Ubuntu 18.04, Pytorch v1.4.0, CUDA 10.0 and cuDNN v7.4.

    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
Install the following Python dependencies (with pip install ):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'
    tqdm
    ninja
    easydict
    argparse
    h5py
    scipy
Compile the CUDA layers , which we used in the network:

    cd pointnet2
    python setup.py install --user

    # Chamfer Distancecd
    cd extensions/chamfer_dist
    python setup.py install --user

    # Cubic Feature Samplingcd
    cd extensions/cubic_feature_sampling
    python setup.py install --user

    # Gridding & Gridding Reversecd
    cd extensions/gridding
    python setup.py install --user

    # apex for multi-gpus train
    cd ./furnace/apex
    export CUDA_HOME=/usr/local/cuda-10.0
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

### Datasets
We use the nyu depth v2, nyucad and suncg_rgbd datasets in our experiments, which are available in ./data folder. The google drive link provides the input data we need (nyu depth v2 and nyucad).

Note that, for suncg_rgbd we follow and obtrain the data from SATNet. https://github.com/ShiceLiu/SATNet

For suncg_rgbd images, please download from the following link and unzip the SUNCGRGBD_images.zip under data/suncg/ https://pan.baidu.com/s/1vUt09GlkC1lPFRm8zXofZA

For the pre-process 2D semantic segmentation, we just use the following semantic
segmentation code (https://github.com/Tramac/awesome-semantic-segmentation-pytorch) which includes many classical methods. So you can train your segmention to obtain the input data of initial scene completion or directly use the npz files provided in the Google Drive (voxel_bisenet_res34_4e-3).

### Get Started
To train SISNet, you can simply use the following command:

    bash scripts/train_SISNet.sh

To test SISNet, you can use the following command:

    bash scripts/test_SISNet.sh

The results are save in results/${dataset}. You can use scripts/toply.py to transform the
npz result to ply and use MeshLab to show.

Finally, the project folder is organized as follows.

    SISNet
    ├── core
    │ ├── scene_completion
    │ ├── instance_completion
    ├── data
    │ ├── nyucad
    │ ├── suncg
    │ ├── nyuv2
    │ │ ├── train.txt
    │ │ ├── test.txt
    │ │ ├── R （RGB images）
    │ │ ├── L (Label ground truth of semantic scene completion)
    │ │ ├── M (Mapping from 2D to 3D)
    │ │ ├── T (TSDF truncated signed distance function)
    │ │ ├── voxel_bisenet_res34_4e-3（sem of visiable region）
    ├── instance_data
    │ ├── nyuv2
    ├── nyuv2_S*_test/train_result
    ├── ckpts
    │ ├── nyuv2_s*.pth
    ├── I*_ckpt
    ├── results
    │ ├── nyuv2
    │ │ ├── *_test/train_results
    ├── utils
    │ ├── vis_ssc.py
    │ ├── ...

### Results on NYU Depth v2 Benchmark
![image](https://github.com/yjcaimeow/SISNet/blob/main/figs/nyu_res.png)
