#!/bin/bash

# TUAB: abnormal classification -------------------------------------------------
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune_tuab_base2 \
#        --log_dir log/finetune_tuab_base2 \
#        --model morgoth_backbone_base \
#        --training_data_dir data/tuh_eeg/TUAB/edf/processed/test \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --epochs 30 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --dataset TUAB \
#        --disable_qkv_bias \
#        --seed 0
# TUAB: abnormal classification -------------------------------------------------


# TUEV: event classification -------------------------------------------------
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune_tuev_base \
#        --log_dir log/finetune_tuev_base2 \
#        --model morgoth_backbone_base \
#        --training_data_dir /data/tuh_eeg/TUEV/edf/processed/test \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --epochs 30 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --dataset TUEV \
#        --disable_qkv_bias \
#        --seed 0
# TUEV: event classification -------------------------------------------------


#TUEP: epilepsy classification -------------------------------------------------
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune_tuep_base \
#        --log_dir log/finetune_tuep_base2 \
#        --model morgoth_backbone_base \
#        --training_data_dir /data/tuh_eeg/TUEP/processed/test \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --epochs 30 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --dataset TUEP \
#        --disable_qkv_bias \
#        --seed 0
#TUEP: epilepsy classification -------------------------------------------------





# NORMAL ----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#          --output_dir checkpoints/finetune6_NORMAL2 \
#          --log_dir log/finetune6_NORMAL2 \
#          --dataset NORMAL \
#          --training_data_dir /data/NORMAL/processed_10second \
#          --model morgoth_backbone_base \
#          --epochs ${EPOCH} \
#          --finetune pretrained_model/base6.pth \
#          --focalloss \
#          --focal_gamma 2 \
#          --focal_alpha "0.5"\
#          --weight_decay 0.05 \
#          --batch_size 64 \
#          --lr 5e-4 \
#          --update_freq 1 \
#          --warmup_epochs 3 \
#          --layer_decay 0.65 \
#          --drop_path 0.1 \
#          --dist_eval \
#          --save_ckpt_freq 1 \
#          --disable_rel_pos_bias \
#          --abs_pos_emb \
#          --disable_qkv_bias \
#          --seed 0
# NORMAL ----------------------------------------------------------------------


# SLOWING ----------------------------------------------------------------------
EPOCH=5
echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
          --output_dir checkpoints/finetune6_SLOWING2 \
          --log_dir log/finetune6_SLOWING2 \
          --dataset SLOWING \
          --training_data_dir /data/SLOWING/processed_10second \
          --focalloss \
          --focal_gamma 2 \
          --focal_alpha "0.25 0.35 0.3" \
          --model morgoth_backbone_base \
          --epochs ${EPOCH} \
          --finetune pretrained_model/base6.pth \
          --weight_decay 0.05 \
          --batch_size 64 \
          --lr 5e-4 \
          --update_freq 1 \
          --warmup_epochs 3 \
          --layer_decay 0.65 \
          --drop_path 0.1 \
          --dist_eval \
          --save_ckpt_freq 1 \
          --disable_rel_pos_bias \
          --abs_pos_emb \
          --disable_qkv_bias \
          --seed 0
# SLOWING ----------------------------------------------------------------------

# BS ----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_BS2 \
#        --log_dir log/finetune6_BS2 \
#        --dataset BS \
#        --training_data_dir /data/BS/processed_10second \
#        --model morgoth_backbone_base \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.5" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
# BS ----------------------------------------------------------------------


# IIIC  ----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_IIIC2 \
#        --log_dir log/finetune6_IIIC2 \
#        --dataset IIIC \
#        --training_data_dir /data/IIIC/processed_10second \
#        --model morgoth_backbone_base \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.2 0.25 0.25 0.25 0.25 0.25" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# IIIC ----------------------------------------------------------------------




# FOC GEN SPIKES 3 class--------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_FOCGENSPIKES2 \
#        --log_dir log/finetune6_FOCGENSPIKES2 \
#        --dataset FOC_GEN_SPIKES \
#        --training_data_dir /data/FOC_GEN_SPIKES/processed_10second \
#        --model morgoth_backbone_base \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.25 0.25 0.25" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# FOC GEN SPIKES 3 class--------------------------------------------------------------


# SPIKES -----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_SPIKE \
#        --log_dir log/finetune6_SPIKE \
#        --dataset SPIKES \
#        --model morgoth_backbone_base \
#        --training_data_dir /data/SPIKES/processed_1second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.5" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# SPIKES -----------------------------------------------------------------------





 # MGB 3 stages SLEEP----------------------------------------------------------------------
# EPOCH=10
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune6_MGBSLEEP3stages \
#         --log_dir log/finetune6_MGBSLEEP3stages \
#         --dataset MGBSLEEP3stages \
#         --training_data_dir /data/MGB_SLEEP/processed_10sec \
#         --model morgoth_backbone_base \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base6.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.25 0.5 0.25" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
 # MGB 3 stages SLEEP ----------------------------------------------------------------------



# SLEEPPSG  ----------------------------------------------------------------------
# EPOCH=10
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune6_SLEEPPSG \
#         --log_dir log/finetune6_SLEEPPSG \
#         --dataset SLEEPPSG \
#         --training_data_dir /data/SLEEP_PSG \
#         --model morgoth_backbone_base \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base6.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.5 0.5 0.5 0.5 0.5" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0


# EPOCH=30
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune_BCHPSG_scaling_focal \
#         --log_dir log/finetune_BCHPSG_scaling_focal \
#         --dataset SLEEPPSG \
#         --training_data_dir /data/SLEEP_PSG/BCH_processed_10sec \
#         --model morgoth_backbone_base \
#         --epochs ${EPOCH} \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.5 0.5 0.5 0.5 0.5" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
#--training_data_dir /data/SLEEP_PSG/processed_30sec \

 # SLEEPPSG ----------------------------------------------------------------------





 # MASS SLEEP----------------------------------------------------------------------
# EPOCH=30
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune3_MASS \
#         --log_dir log/finetune3_MASS \
#         --dataset SLEEPMASS \
#         --training_data_dir /data/MASS/processed_10sec \
#         --model morgoth_backbone_base \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base3.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.5 0.5 0.5 0.5 0.5" \
#         --weight_decay 0.05 \
#         --batch_size 128 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
 # MASS SLEEP ----------------------------------------------------------------------


 # PENN SLEEP----------------------------------------------------------------------
# EPOCH=10
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune3_PENN_nofinetune \
#         --log_dir log/finetune3_PENN_nofinetune \
#         --dataset SLEEPPENN \
#         --training_data_dir /data/PENN/processed_10second \
#         --model morgoth_backbone_base \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base3.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.25 0.5 0.25 0.25 0.4" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
 # PENN SLEEP----------------------------------------------------------------------







# FOC SPIKES --------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_FOCSPIKES \
#        --log_dir log/finetune3_FOCSPIKES \
#        --dataset FOC_SPIKES \
#        --training_data_dir /data/FOC_GEN_SPIKES/FOC_NO/processed_10second \
#        --model morgoth_backbone_base \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
#
#echo "exxact@1" | sudo mkdir checkpoints/finetune3_FOCSPIKES_Occasion
#echo "exxact@1" | sudo cp checkpoints/finetune3_FOCSPIKES/checkpoint-best.pth checkpoints/finetune3_FOCSPIKES_Occasion/checkpoint-0.pth
#EPOCH=3
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_FOCSPIKES_Occasion \
#        --log_dir log/finetune3_FOCSPIKES_Occasion \
#        --dataset GEN_SPIKES \
#        --training_data_dir /data/OccasionNoise/proccessed_10second/fs_train \
#        --model morgoth_backbone_base \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# FOC SPIKES--------------------------------------------------------------


# GEN SPIKES --------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_GENSPIKES \
#        --log_dir log/finetune3_GENSPIKES \
#        --dataset GEN_SPIKES \
#        --training_data_dir /data/FOC_GEN_SPIKES/GEN_NO/processed_10second \
#        --model morgoth_backbone_base \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

#echo "exxact@1" | sudo mkdir /home/exx/Documents/EEG_report/EEGfounder/checkpoints/finetune3_GENSPIKES_Occasion
#echo "exxact@1" | sudo cp /home/exx/Documents/EEG_report/EEGfounder/checkpoints/finetune3_GENSPIKES/checkpoint-best.pth /home/exx/Documents/EEG_report/EEGfounder/checkpoints/finetune3_GENSPIKES_Occasion/checkpoint-0.pth
#EPOCH=3
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_GENSPIKES_Occasion \
#        --log_dir log/finetune3_GENSPIKES_Occasion \
#        --dataset GEN_SPIKES \
#        --training_data_dir /data/OccasionNoise/proccessed_10second/gs_train \
#        --model morgoth_backbone_base \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
# GEN SPIKES--------------------------------------------------------------