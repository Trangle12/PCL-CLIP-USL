CUDA_VISIBLE_DEVICES=0 python train.py -b 256 -d msmt17 \
--iters 200 --eps 0.7 --logs-dir ./log/msmt17

CUDA_VISIBLE_DEVICES=0 python train.py -b 256 -d market1501 \
--iters 200 --eps 0.5 --logs-dir ./log/market1501