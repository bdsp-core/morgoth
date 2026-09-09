#!/bin/bash

## Required parameters
#--dataset IIIC or SPIKES or FOC_GEN_SPIKES or BS or SLOWING or NORMAL or SLEEPPSG or MGBSLEEP3stages
#--data_format edf or mat
#--eval_sub_dir /xxx/xxx/ (input data dir)
#--eval_results_dir /xxx/xxx/ (output result dir)
#--prediction_slipping_step xxx (Step size in points; if original hz>200 prediction_slipping_step better to be 100 or 128)
#  or use --prediction_slipping_step_second xxx (step size in seconds)

##### Optional parameters: If the original data contains channel names and sampling rate information, the following parameters can be omitted from the command.
#--sampling_rate 0 or xxx (If the raw data does not contain this information, it should be assigned here; 0 indicates that the information is present in the data.)
#--already_format_channel_order yes (If the data does not include channel information, it needs to be sorted as required before being input.)
#--already_average_montage yes (If the data has already been average montaged, it should be specified.)
#--allow_missing_channels yes or no (If the data does not include all 19 channels, processing is still allowed — the missing channels will be zero-filled.)

##### Optional parameters: For 1-second spike detection
#--smooth_result ema or window_ema or ''
#--need_spikes_10s_result yes (summarize 10-second results from 1-second predictions.)
#--spikes_10s_result_slipping_step_second xx (sliding step in second for 10-second spike detection)

##### More optional parameters:
#--polarity 1 or -1 with default 1 (If set -1, the signal is inverted)
#--max_length_hour no or 1,2,3...(Only analyze the first n hours of the EEG)
#--leave_one_hemisphere_out no or left or right or middel (Set the EEG signals to 0 for the left, right, or middle hemisphere)
#--rewrite_results no | yes (Default no Overwrite the original results when new results are available.)

password="exxact@1"

# 1. IIIC-------------------------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model morgoth_backbone_base \
            --predict \
            --task_model checkpoints/IIIC.pth \
            --dataset IIIC \
            --data_format mat \
            --sampling_rate 200 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
            --max_length_hour no \
            --leave_one_hemisphere_out no \
            --polarity 1 \
            --eval_sub_dir test_data/IIIC/segments_raw \
            --eval_results_dir test_data/IIIC/results/pred_1sStep\
            --prediction_slipping_step_second 1 \
            --rewrite_results no



# 2. SPIKES--------------------------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model morgoth_backbone_base \
            --predict \
            --task_model checkpoints/SPIKES.pth \
            --dataset SPIKES \
            --data_format mat \
            --sampling_rate 128 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
            --max_length_hour no \
            --leave_one_hemisphere_out no \
            --polarity 1 \
            --eval_sub_dir test_data/SPIKES/SN2/segments_10min \
            --eval_results_dir test_data/SPIKES/SN2/results/pred_1pStep \
            --prediction_slipping_step 1 \
            --smooth_result ema

echo "$password" |  sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
           --abs_pos_emb \
            --model morgoth_backbone_base \
            --predict \
            --task_model checkpoints/SPIKES.pth \
            --dataset SPIKES \
            --data_format edf \
            --sampling_rate 0 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
            --max_length_hour no \
            --leave_one_hemisphere_out no \
            --polarity 1 \
            --eval_sub_dir test_data/SPIKES/HEP/edf \
            --eval_results_dir test_data/SPIKES/HEP/results/pred_1pStep \
            --prediction_slipping_step 1 \
            --smooth_result ema \
            --need_spikes_10s_result yes \
            --spikes_10s_result_slipping_step_second 1


# 3. Focal/Generalized Spikes--------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model morgoth_backbone_base \
            --predict \
            --task_model checkpoints/FOCGENSPIKES.pth \
            --dataset FOC_GEN_SPIKES \
            --data_format mat \
            --sampling_rate 100 \
            --already_format_channel_order no \
            --already_average_montage yes \
            --allow_missing_channels no \
            --max_length_hour no \
            --polarity 1 \
            --leave_one_hemisphere_out no \
            --eval_sub_dir test_data/MoE_event/mat \
            --eval_results_dir test_data/MoE_event/results/pred_FOCGENSPIKES_1sStep \
            --prediction_slipping_step_second 1


# 4. Slowing--------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model morgoth_backbone_base \
            --predict \
            --task_model checkpoints/SLOWING.pth \
            --dataset SLOWING \
            --data_format mat \
            --sampling_rate 100 \
            --already_format_channel_order no \
            --already_average_montage yes \
            --allow_missing_channels no \
            --max_length_hour no \
            --polarity 1 \
            --leave_one_hemisphere_out no \
            --eval_sub_dir test_data/SLOWING/segments_10min \
            --eval_results_dir test_data/SLOWING/results/pred_1sStep \
            --prediction_slipping_step_second 1


# 5. BS--------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model morgoth_backbone_base \
            --predict \
            --task_model checkpoints/BS.pth \
            --dataset BS \
            --data_format mat \
            --sampling_rate 100 \
            --already_format_channel_order no \
            --already_average_montage yes \
            --allow_missing_channels no \
            --max_length_hour no \
            --leave_one_hemisphere_out no \
            --polarity 1 \
            --eval_sub_dir test_data/MoE_event/mat \
            --eval_results_dir test_data/MoE_events/results/pred_BS_1sStep \
            --prediction_slipping_step_second 1


# 6. NORMAL--------------------------------------------------------
echo "$password" | sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model morgoth_backbone_base \
            --predict \
            --task_model checkpoints/NORMAL.pth \
            --dataset NORMAL \
            --data_format edf \
            --sampling_rate 0 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
            --max_length_hour no \
            --polarity 1 \
            --leave_one_hemisphere_out no \
            --eval_sub_dir test_data/Sandor/EDF \
            --eval_results_dir test_data/Sandor/results/pred_NORMAL_1sStep \
            --prediction_slipping_step_second 1


# 7. SLEEP 3 stages with 19 channels --------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/SLEEP.pth \
            --abs_pos_emb \
            --dataset MGBSLEEP3stages \
            --data_format mat \
            --sampling_rate 100 \
            --already_format_channel_order no \
            --already_average_montage yes \
            --allow_missing_channels no \
            --max_length_hour no \
            --polarity 1 \
            --leave_one_hemisphere_out no \
            --eval_sub_dir test_data/MoE_event/mat \
            --eval_results_dir test_data/MoE_event/results/pred_SLEEP3Stages_1sStep \
            --prediction_slipping_step_second 1


# 8. SLEEP 5 stages with 6 channels --------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/SLEEPPSG.pth \
            --abs_pos_emb \
            --dataset SLEEPPSG \
            --data_format mat \
            --sampling_rate 100 \
            --already_format_channel_order no \
            --already_average_montage yes \
            --allow_missing_channels no \
            --max_length_hour no \
            --polarity 1 \
            --leave_one_hemisphere_out no \
            --eval_sub_dir test_data/MoE_event/mat \
            --eval_results_dir test_data/MoE_event/results/pred_SLEEP5Stages_1sStep \
            --prediction_slipping_step_second 1


######################################## IF USING CPU #######################################
# if you do not have gpu, to use cpu by using python command and adding "-device cpu" and "--distributed False" as follow
#echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
#            --abs_pos_emb \
#            --model morgoth_backbone_base \
#            --predict \
#            --task_model checkpoints/IIIC.pth \
#            --dataset IIIC \
#            --data_format mat \
#            --sampling_rate 200 \
#            --already_format_channel_order no \
#            --already_average_montage no \
#            --allow_missing_channels no \
#            --max_length_hour no \
#            --leave_one_hemisphere_out no \
#            --polarity 1 \
#            --eval_sub_dir test_data/IIIC/segments_raw \
#            --eval_results_dir test_data/IIIC/results/pred_1sStep \
#            --prediction_slipping_step_second 1 \
#            --device cpu \
#            --distributed False
######################################## IF USING CPU #######################################