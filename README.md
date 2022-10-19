## ROME 2.0 Instance Segmentation

This repository contains the instance segmentation network architecture and training loop that uses detectron2


#### TODO
- 

## Installation

Please make sure you have a CUDA enabled NVIDIA GPU and CUDA version 11.3 or above.

### Clone repository
```
cd ias
```

### Conda setup

Install Conda
```
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh
source ~/.bashrc 
```

Create conda environment
```
conda env create -f environment.yml
```

**NOTE: Environment takes several minutes (up to ten) to resolve. This is because of cudatoolkit. However, the environment will resolve. DO NOT QUIT WHILE ENVIRONMENT IS RESOLVING!**

Activate environment
```
conda activate ias
```

Install detectron2 from [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). 

Alternately, the command below works as of July 14 2022.
```
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
git checkout v0.5
cd ..
python -m pip install -e detectron2
```

## Download Models

### Download IAS model
```
mkdir -p models && cd models
wget https://tu-dortmund.sciebo.de/s/ISdLcDMduHeW1ay/download  -O FAT_trained_Ml2R_bin_fine_tuned.pth
```
 
 ### Download baseline mask-rcnn model
 ```
mkdir -p models/mask-rcnn && cd models/mask-rcnn
wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
 ```

## Hyperparameter Tuning Results

Results can be found in `test_results.txt`. From left to right the results contain dataset used for training, specific hyperparameters for that training and finally the AP.
See `train_net.py` for more information on hyperparameter usage.

## Run inference 
```
cd ../scripts
python segment_image.py --image-path /path/to/image
```

## Downloading Datasets

### NVIDIA FAlling Things Dataset

Visit [here](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation) for dataset download instructions.

Alternately,

```
pip install gdown
cd /path/to/datasets
gdown 1y4h9T6D9rf6dAmsRwEtfzJdcghCnI_01
unzip fat.zip
```
The dataset is 40+ GB and will take a while to download.

Create coco annotations for the dataset. (See `create_coco_json.py` for more information)
```
cd ../scripts
python create_coco_json.py --dataset nvidia --data-path /path/to/dataset
```

### DoPose dataset

Visit [here](https://zenodo.org/record/6103779#.YtHUytLMJhE) for dataset download instructions.

Alternately,

```
cd /path/to/datasets
mkdir doPose && cd doPose
wget https://zenodo.org/record/6103779/files/test_bin.zip
wget https://zenodo.org/record/6103779/files/test_table.zip
unzip test_bin.zip
unzip test_table.zip
```
Create coco annotations for the dataset. (See `create_coco_json.py` for more information)

```
cd ../../scripts
python create_coco_json.py --dataset dopose --data-path /path/to/dataset
```

### SPS dataset

#### Synthetic SPS dataset using blenderproc

Download and install blenderproc (follow instructions from repo)

```
git clone git@github.com:ajonnavittula/BlenderProc.git
```

Edit data-path in `run.sh` and then run using
```
cd BlenderProc
bash run.sh
```

This will create the dataset in appropriate format. To generate COCO annotations run:
```
cd /location/to/ias/scripts
python create_coco_json.py --dataset sps --data-path /path/to/dataset
```

## Train models
Following the instructions from [here](https://arxiv.org/pdf/2204.13613.pdf) we train on NVIDIA FAT dataset first and then the DoPose dataset. The following instructions will train according to the guidelines from the paper.

### Training on NVIDIA FAT dataset
Since the dataset is too large for training at once, create a JSON with chunks
```
python create_nvidia_fat_chunks.py
```

Train the network.
```
python train_net.py --dataset nvidia --data-path /path/to/dataset --initial \
--max-iter 1700 --lr 0.00001 --num-gpus 2
```

See `train_net.py` for more information on arguments.

### Training on DoPose dataset
```
python train_net.py --dataset dopose --data-path /path/to/dataset \
--max-iter 500 --lr 0.0000001 --num-gpus 2
```

See `train_net.py` for more information on arguments.


### Training on SPS dataset

```
python train_net.py --dataset sps --data-path /path/to/dataset \
--max-iter 500 --lr 0.0000001 --num-gpus 2
```

## Evaluating Model Performance
```
python evaluate_net --model-path /path/to/model
```
Assumes that the dataset to be evaluated is sps in `../datasets/sps`, with the json file in `../datasets/sps/sps.json`.


## Exporting trained Model to C++

### Prerequisites

Download / install the following C++ libraries. 

- gcc ---- v9.x
- cudatoolkit ---- v11.3
- cudnn ---- v8.x
- libtorch ---- v1.10.1
- opencv ---- v3.4.12 (GPU compiled)
- torchvision ---- v0.11.2
- detectron2 ---- v0.5

### Export model to torchscript

Change `--sample-image` and `--data-path` to a valid paths and run the following commands.

```
cd /path/to/ias
cd scripts
python export_torchscript.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --sample-image /path/to/image --data-path /path/to/datasets --export-method tracing --format torchscript   MODEL.WEIGHTS ../models/FAT_trained_Ml2R_bin_fine_tuned.pth     MODEL.DEVICE cuda
```

The exported model will be saved in `../export`

### Build detectron2 C++ code

Comment out executable for Caffe2 from CMakeLists.txt and build

```
cd path/to/detectron2/tools/deploy
mkdir build && cd build
cmake ..
make
```

Follow instructions in README.md to test exported model.

## Post Processing Instructions

### Generating Binary Masks
```
python create_masks --image-path /path/to/images --model-path /path/to/model
```
By default the image path is `../datasets/sps/rgb/`. This will create binary masks for every detected instance (one instance per image), and save them into `../datasets/sps/masks/`.

### Reformatting Masks for Post-Processing
```
python arrange_for_postprocessing
```
Will create all required directories and name the masks appropriately in `../postprocessing/`. Requires you to run the `create_masks` script first.

### Instructions for Building Post-Processing Code
```
cd instance-proc/
mkdir -p build && cd build
cmake ..
make
```
Will create executables inside of the `/build/` directory.

### Create PCL instances
```
cd .. 
mkdir -p dataset
mv ../postprocessing/ /dataset/
cd build
./instance-cloud
```
Will generate PCL clouds for the instance segmentation output. Currently assumes presence of postprocessing data in the original ias directory. Will create clouds for all the data present.

### Generate Pick Points
```
./pcl-proc 1
```
If you don't want to visualize the pick points just replace 1 with 0.
Still adding some utility functions to facilitate data manipulation. Maybe allow use to select image name.

