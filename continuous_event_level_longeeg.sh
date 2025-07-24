password="exxact@1"


echo "$password" | sudo -S  $(which python) segment_long_eeg.py segment \
            --data_format edf  \
            --segment_duration 600 \
            --eeg_dir test_data/longEEG/raw   \
            --eval_sub_dir test_data/longEEG/segments_raw  \


# IIIC-------------------------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
            --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model checkpoints/IIIC.pth \
            --dataset IIIC \
            --data_format edf \
            --sampling_rate 0 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
             --max_length_hour no \
            --eval_sub_dir  test_data/longEEG/segments_raw \
            --eval_results_dir test_data/longEEG/results/IIIC_pred_1sStep\
            --prediction_slipping_step_second 1 \
            --rewrite_results no


# Spike--------------------------------------------------------
echo "$password" |  sudo -S OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=3 finetune_classification.py \
           --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model checkpoints/SPIKES.pth \
            --dataset SPIKES \
            --data_format edf \
            --sampling_rate 0 \
            --already_format_channel_order no \
            --already_average_montage no \
            --allow_missing_channels no \
            --max_length_hour no \
            --eval_sub_dir test_data/longEEG/segments_raw  \
            --eval_results_dir test_data/longEEG/results/Spike_pred_32pStep\
            --prediction_slipping_step 32 \
            --smooth_result ema \
            --rewrite_results no


# Slowing--------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12346 finetune_classification.py \
            --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model checkpoints/SLOWING.pth \
            --dataset SLOWING \
            --data_format edf \
            --sampling_rate 0 \
            --already_format_channel_order no \
            --already_average_montage yes \
            --allow_missing_channels no \
            --max_length_hour no \
            --eval_sub_dir test_data/longEEG/segments_raw \
            --eval_results_dir test_data/longEEG/results/Slowing_pred_1sStep\
            --prediction_slipping_step_second 1 \
            --rewrite_results no



# BS--------------------------------------------------------
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12346 finetune_classification.py \
            --abs_pos_emb \
            --model base_patch200_200 \
            --predict \
            --task_model checkpoints/BS.pth \
            --dataset BS \
            --data_format edf \
            --sampling_rate 0 \
            --already_format_channel_order no \
            --already_average_montage yes \
            --allow_missing_channels no \
            --max_length_hour no \
            --eval_sub_dir test_data/longEEG/segments_raw \
            --eval_results_dir test_data/longEEG/results/BS_pred_1sStep\
            --prediction_slipping_step_second 1 \
            --rewrite_results no



