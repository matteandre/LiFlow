# LiFlow: Flow Matching for 3D LiDAR Scene Completion

![](media/LiFlow.png)

## Dependencies

Installing python and pre-requisites packages with Anaconda:

`conda create -n liflow python=3.9.21`

`conda activate liflow`

`conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia`

`conda install openblas-devel -c anaconda`

`conda install cuda-toolkit -c nvidia/label/cuda-11.8.0`

`conda install cudatoolkit==11.8 -c pytorch`

`conda install anaconda::cython`



Installing MinkowskiEngine:

`export CUDA_HOME=$CONDA_PREFIX`

```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Installing pytorch3D:

`conda install -c iopath iopath`

`pip install "git+https://github.com/facebookresearch/pytorch3d.git"`

Installing other dependencies on the code main directory:

```
cd LiFlow
pip install -r requirements.txt
```


To setup the code run the following command on the code main directory:

```
cd LiFlow
pip install -U -e .
```


## The SemanticKITTI Dataset

The SemanticKITTI dataset has to be download from [site](http://www.semantic-kitti.org/dataset.html#download) and extracted in the following structure:

```
./liflow/
└── Datasets/
    └── SemanticKITTI
        └── dataset
          └── sequences
            ├── 00/
            │   ├── velodyne/
            |   |       ├── 000000.bin
            |   |       ├── 000001.bin
            |   |       └── ...
            │   └── labels/
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```

## The Apollo Dataset

The Apollo dataset can be downloaded from [site](https://www.ipb.uni-bonn.de/html/projects/apollo_dataset/LiDAR-MOS.zip) and extracted in the following structure:

```
./liflow/
└── Datasets/
    └── LiDAR-MOS
          └── sequences
            ├── 00/	# for validation
            │   ├── velodyne/
            |   |       ├── 000000.bin
            |   |       ├── 000001.bin
            |   |       └── ...
            │   └── labels/
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            └──  04/
                 └── ...
```

## Ground truth generation

To generate the ground complete scenes you can run the `map_from_scans.py` script. This will use the dataset scans and poses to generate the sequence map to be used as ground truth during training:

```
python utils/map_from_scans.py --path ./Datasets/SemanticKITTI/dataset/sequences
```

Once the sequences map is generated you can then train the model

## Training the LiFlow model

For training the LiFlow model, the configurations are defined in `config/config_flow.yaml`, and the training can be started with:

`python train.py`


## Evaluate Flow Scene Completion

For running the scene completion evaluation:

`python utils/eval_path.py --path path/to/data --flow flow_ckpt --refine refine_ckpt  `

For generating the scene completion point clouds:

`python utils/flow_completion_pipeline.py --path path/to/data --flow flow_ckpt --refine refine_ckpt  `


## Pre-trained weights

Pre-trained weights are avilable in [site](https://drive.google.com/file/d/1uvQeDrajY_qWdSzX4PDvMdMkBLmEJ570/view?usp=sharing)


## References

The refinement network is provided from [site](https://github.com/PRBonn/LiDiff)

## Citation

```
@inproceedings{matteazzi2026liflow,
  title={Liflow: Flow matching for 3d lidar scene completion},
  author={Matteazzi, Andrea and Tutsch, Dietmar},
  booktitle={European Conference on Computer Vision},
  pages={130--144},
  year={2026},
  organization={Springer}
}
```
