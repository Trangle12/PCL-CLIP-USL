# [NOT Official] Prototypical Contrastive Learning-based CLIP Fine-tuning for Unsupervised Person Re-Identification

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

## Training

For example, training the full model on Market1501 with GPU 0 and saving the log file and checkpoints to `logs/market-pclclip`:

```
CUDA_VISIBLE_DEVICES=0 python train.py -b 256 -d market1501 --iters 200 --eps 0.4 --use-hard --logs-dir ./log/market1501

CUDA_VISIBLE_DEVICES=0 python train.py -b 256 -d msmt17 --iters 200 --eps 0.7 --logs-dir ./log/msmt
```

## Results

The results are on Market1501 (M) and MSMT17 (MS). 
| Methods | M | Link | MS | Link |
| --- | -- | -- | -- | - |
| CC + PCL-CLIP | 86.9 (94.2) | - | 56.4 (77.9) | - |
| CC + PCL-CLIP (Reproduce) | 90.2 (95.5) | [model](https://drive.google.com/file/d/1Qvz7YNEVBjzDcPSE_1Ks6nuj_yAmEMVW/view?usp=sharing) | 67.5 (85.1) | [model](https://drive.google.com/file/d/1Q-QC9bTuf_iA7rtoot3QI8Z-gko1RnoO/view?usp=sharing) |

## Note
The model training requires the graphics card to be greater than 20GB. In this experiment, I used a single NVIDIA A100 with a memory of 40GB to carry out the relevant work. 

The code is implemented based on following works.

1. [PCL-CLIP](https://github.com/RikoLi/PCL-CLIP)
2. [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid)



