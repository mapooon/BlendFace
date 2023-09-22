# Training BlendFace on Your Machine
Here, we provide the training code for our encoder BlendFace.

# Preparation
## Training Data
Please download `ms1m-retinaface-t1` dataset following [InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) and place it in `data/`.

## Precomputed Data
We provide [precomputed data](https://drive.google.com/drive/folders/1ik_pX1ZFK4YCSc-QulwUCTV8YP528xlj?usp=sharing) (segmentation masks, nearest sample list, etc...) to generate blended images during training.
Please download all the files and place them in `data/ms1m-retinaface-t1/`.  
Then, unzip masks.tar.gz:
```bash
tar -zxvf masks.tar.gz
```

## Dataset Directory
```
training
└── data
    └── ms1m-retinaface-t1
        ├── train.idx
        ├── train.rec
        ├── idx2id.json
        ├── idx2label.json
        ├── idx2nearestidxs.pkl
        └── masks
```

## Docker
(1) Pull a docker image from docker hub:
```bash
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
```
(2) Replace the absolute path to this repository in `./exec.sh`.  
(3) Execute the image:
```bash
bash exec.sh
```
(4) Install packages:
```bash
bash install.sh
```


# Training
Train BlendFace on four NVIDIA A100 GPUs.
```shell
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/ms1mv3_r100_aug
```
The results are saved in `work_dirs/`.

# Acknowledgements
Our code is heavily based on [InsightFace](https://github.com/deepinsight/insightface).


# Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{shiohara2023blendface,
  title={BlendFace: Re-designing Identity Encoders for Face-Swapping},
  author={Shiohara, Kaede and Yang, Xingchao and Taketomi, Takafumi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```