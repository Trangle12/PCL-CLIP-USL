# [NOT official] Prototypical Contrastive Learning-based CLIP Fine-tuning for Unsupervised Person Re-Identification

[[paper]](https://arxiv.org/pdf/2310.17218.pdf)

## Upload History

* 2025/03/03: Unsupervised Code.

## Installation

Install `conda` before installing any requirements.

```bash
conda create -n pclclip python=3.9
conda activate pclclip
pip install -r requirements.txt
```

## Datasets

Make a new folder named `data` under the root directory. Download the datasets and unzip them into `data` folder.
* [Market1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
* [MSMT17](https://arxiv.org/abs/1711.08565)
* [VeRi776](https://github.com/JDAI-CV/VeRidataset)

## Training

For example, training the full model on Market1501 with GPU 0 and saving the log file and checkpoints to `logs/market-pclclip`:

```
CUDA_VISIBLE_DEVICES=3 python train.py -b 256 -d market1501 --iters 200 --eps 0.5 --logs-dir ../log/market1501

CUDA_VISIBLE_DEVICES=2 python train.py -b 256 -d msmt17 --iters 200 --eps 0.7 --logs-dir ../log/msmt
```

## Note

The code is implemented based on following works.

1. [TransReID](https://github.com/damo-cv/TransReID)
2. [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID)
3. [PCL-CLIP](https://github.com/RikoLi/PCL-CLIP)
4. [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid)

