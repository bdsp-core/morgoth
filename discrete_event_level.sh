#!/bin/bash

# Each data should be saved as a single mat/pkl file

# Data should be 19 channels 'FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2' (6 channels 'F3', 'C3',  'O1',  'F4', 'C4', 'O2' for SLEEPPSG task)
#   -> 200hz
#   -> 0.5-70 bandpass
#   -> 50 and 60 hz notch
#   -> common average
#   -> clip at +-500 uv
#   -> DO NOT normalization
#   -> 10s segments(1s for SPIKES task)
# Mat file should be {'data': X, 'y': y}. X is data, y is its label 0/1/2/3/4/5
# See cases in test_data/SPIKES/processed_1second or test_data/IIIC/processed_10second

# nb_classes: SPIKES/NORMAL/BS 1; IIIC 6; SLOWING/FOC_GEN_SPIKES/MGBSLEEP3stages 3; SLEEPPSG 5
# Output results will be saved in pred.csv in eval_sub_dir folder

password="exxact@1"

# 1. Spikes
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --eval \
            --model morgoth_backbone_base \
            --task_model checkpoints/SPIKES.pth \
            --dataset SPIKES \
            --nb_classes 1 \
            --test_data_format mat \
            --eval_sub_dir test_data/SPIKES/processed_1second



# 2. Seizure+IIIC
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --eval \
            --model morgoth_backbone_base \
            --task_model checkpoints/IIIC.pth \
            --dataset IIIC \
            --nb_classes 6 \
            --test_data_format mat \
            --eval_sub_dir test_data/IIIC/processed_10second



# 3. Focal Generalized Slowing
#echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model morgoth_backbone_base \
#            --task_model checkpoints/SLOWING.pth \
#            --dataset SLOWING \
#            --nb_classes 3 \
#            --test_data_format mat \
#            --eval_sub_dir test_data/SLOWING/processed_10second


## 4. Focal Generalized Spikes
#echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model morgoth_backbone_base \
#            --task_model checkpoints/FOC_GEN_SPIKES.pth \
#            --dataset FOC_GEN_SPIKES \
#            --nb_classes 3 \
#            --test_data_format mat \
#            --eval_sub_dir test_data/FOC_GEN_SPIKES/processed_10second
#
#
#
## 5. Burst Suppression
#echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port= finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model morgoth_backbone_base \
#            --task_model checkpoints/BS.pth \
#            --dataset BS \
#            --nb_classes 1 \
#            --test_data_format mat \
#            --eval_sub_dir test_data/BS/processed_10second
#
#
#
## 6. Normal
#echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model morgoth_backbone_base \
#            --task_model checkpoints/BS.pth \
#            --dataset NORMAL \
#            --nb_classes 1 \
#            --test_data_format mat \
#            --eval_sub_dir test_data/NORMAL/processed_10second
#
#
#
## 7. Sleep staging (5 stages; input 6 EEG channels)
#echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model morgoth_backbone_base \
#            --task_model checkpoints/SLEEPPSG.pth \
#            --dataset SLEEPPSG \
#            --nb_classes 5 \
#            --test_data_format mat \
#            --eval_sub_dir test_data/SLEEPPSG/processed_10second
#
#
#
## 8. Sleep staging (3 stages; input 19 EEG channels)
#echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#            --abs_pos_emb \
#            --eval \
#            --model morgoth_backbone_base \
#            --task_model checkpoints/MGBSLEEP3stages.pth \
#            --dataset MGBSLEEP3stages \
#            --nb_classes 3 \
#            --test_data_format mat \
#            --eval_sub_dir test_data/MGBSLEEP3stages/processed_10second
#
#
