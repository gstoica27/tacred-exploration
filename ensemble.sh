#!/bin/bash

# This is an example script of training and running model ensembles.

# train 5 models with different seeds
CUDA_VISIBLE_DEVICES=3 python train.py --seed 1234 --data_dir /home/scratch/gis/datasets/tacred/data/json/ --vocab_dir /home/scratch/gis/datasets/tacred-relation_data/ --id 01 --info "Position-aware attention model" --optim adagrad
CUDA_VISIBLE_DEVICES=3 python train.py --seed 3145 --data_dir /home/scratch/gis/datasets/tacred/data/json/ --vocab_dir /home/scratch/gis/datasets/tacred-relation_data/ --id 02 --info "Position-aware attention model" --optim adagrad
CUDA_VISIBLE_DEVICES=3 python train.py --seed 1776 --data_dir /home/scratch/gis/datasets/tacred/data/json/ --vocab_dir /home/scratch/gis/datasets/tacred-relation_data/ --id 03 --info "Position-aware attention model" --optim adagrad
CUDA_VISIBLE_DEVICES=3 python train.py --seed 2019 --data_dir /home/scratch/gis/datasets/tacred/data/json/ --vocab_dir /home/scratch/gis/datasets/tacred-relation_data/ --id 04 --info "Position-aware attention model" --optim adagrad
CUDA_VISIBLE_DEVICES=3 python train.py --seed 420 --data_dir /home/scratch/gis/datasets/tacred/data/json/ --vocab_dir /home/scratch/gis/datasets/tacred-relation_data/ --id 05 --info "Position-aware attention model" --optim adagrad

# evaluate on test sets and save prediction files
CUDA_VISIBLE_DEVICES=3 python eval.py /home/scratch/gis/datasets/tacred-relation_data/saved_models/01 --out /home/scratch/gis/datasets/tacred-relation_data/saved_models/out/test_1.pkl
CUDA_VISIBLE_DEVICES=3 python eval.py /home/scratch/gis/datasets/tacred-relation_data/saved_models/02 --out /home/scratch/gis/datasets/tacred-relation_data/saved_models/out/test_2.pkl
CUDA_VISIBLE_DEVICES=3 python eval.py /home/scratch/gis/datasets/tacred-relation_data/saved_models/03 --out /home/scratch/gis/datasets/tacred-relation_data/saved_models/out/test_3.pkl
CUDA_VISIBLE_DEVICES=3 python eval.py /home/scratch/gis/datasets/tacred-relation_data/saved_models/04 --out /home/scratch/gis/datasets/tacred-relation_data/saved_models/out/test_4.pkl
CUDA_VISIBLE_DEVICES=3 python eval.py /home/scratch/gis/datasets/tacred-relation_data/saved_models/05 --out /home/scratch/gis/datasets/tacred-relation_data/saved_models/out/test_5.pkl

# run ensemble
ARGS="--data_dir /home/scratch/gis/datasets/tacred/data/json/"
for id in 1 2 3 4; do
    OUT="/home/scratch/gis/datasets/tacred-relation_data/saved_models/out/test_${id}.pkl"
    ARGS="$ARGS $OUT"
done
python ensemble.py --dataset test $ARGS

