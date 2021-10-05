# MLMO
This repository contains the Python-Chainer implementation of the following paper:

**Deep Metric Learning for Multi-Label and Multi-Object Image Retrieval**,  
Jonathan Mojoo, Takio Kurita

**IEICE Transactions on Information and Systems** 2019 ([PDF](https://www.researchgate.net/profile/Takio-Kurita/publication/352034949_Deep_Metric_Learning_for_Multi-Label_and_Multi-Object_Image_Retrieval/links/60bf03a0a6fdcc22eae8ca4a/Deep-Metric-Learning-for-Multi-Label-and-Multi-Object-Image-Retrieval.pdf))

If you use this code in your research, please cite:
```
@article{mojoo2021deep,
  title={Deep Metric Learning for Multi-Label and Multi-Object Image Retrieval},
  author={Mojoo, Jonathan and Kurita, Takio},
  journal={IEICE Transactions on Information and Systems},
  volume={104},
  number={6},
  pages={873--880},
  year={2021},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}
```

## Setup
* Install Chainer with CuPy (For GPU support)
* Put the path to your image data in ```./data/<dataset-name>/data-dir.txt```
* Download pre-trained weights in ```.npz``` format

## Training
### NUSWIDE
* ```python train_net.py --dataset nuswide --extractor alexnet --pretrained <path-to-pretrained-model> --gpu 0 --out results/nuswide --stage 2```
### MSCOCO (with VGG-16)
* ```python train_net.py --dataset nuswide --extractor vgg16 --gpu 0 --out results/nuswide --stage 2```
