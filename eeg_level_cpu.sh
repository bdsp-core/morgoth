# if you do not have gpu, to use cpu by using python command and adding "-device cpu" and "--distributed False" as follow

password="exxact@1"

# Sandor-100 case
dataset_dir="test_data/Sandor/EDF"
data_format="edf"
sampling_rate=0
result_dir="test_data/Sandor/results"
already_format_channel_order='no'
already_average_montage='no'
allow_missing_channels='no'
leave_one_hemisphere_out='no'
polarity=-1  # Sandor EEG EDF file has a polarity flip.
rewrite_results='no'

#  Normal:
     # (1). Continuous 1-second step event-level prediction

echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/NORMAL.pth \
            --abs_pos_emb \
            --dataset NORMAL \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_NORMAL_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}

    # (2). EEG_level prediction
echo "$password" | sudo -S  $(which python) EEG_level_head.py \
        --mode predict \
        --dataset NORMAL \
        --task_model checkpoints/NORMAL_EEGlevel.pth \
        --test_csv_dir test_data/Sandor/pred_NORMAL_1sStep \
        --result_dir test_data/Sandor \
        --device cpu \


#  Slowing:
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/SLOWING.pth \
            --abs_pos_emb \
            --dataset SLOWING \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order}  \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLOWING_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}

    # (2). EEG_level prediction (FOC_SLOWING and GEN_SLOWING are seperated)
SLOWING_datasets=("FOC_SLOWING" "GEN_SLOWING")
for SLOWING_dataset in "${SLOWING_datasets[@]}"; do
    echo "$password" | sudo -S  $(which python) EEG_level_head.py \
            --mode predict \
            --dataset ${SLOWING_dataset} \
            --task_model checkpoints/${SLOWING_dataset}_EEGlevel.pth \
            --test_csv_dir ${result_dir}/pred_SLOWING_1sStep \
            --result_dir ${result_dir}
done


# BS
  # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/BS.pth \
            --abs_pos_emb \
            --dataset BS \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_BS_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}


  # (2). EEG_level prediction
echo "$password" | sudo -S  $(which python) EEG_level_head.py \
          --mode predict \
          --dataset BS \
          --task_model checkpoints/BS_EEGlevel.pth \
          --test_csv_dir ${result_dir}/pred_BS_1sStep  \
          --result_dir ${result_dir}



# FOC GEN SPIKES:
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/FOCGENSPIKES.pth \
            --abs_pos_emb \
            --dataset FOC_GEN_SPIKES \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir  ${result_dir}/pred_FOCGENSPIKES_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}


    # (2). EEG_level prediction (FOC_SPIKES and GEN_SPIKES are seperated)
FOC_GEN_SPIKES_datasets=("FOC_SPIKES" "GEN_SPIKES")
for FOC_GEN_SPIKES_dataset in "${FOC_GEN_SPIKES_datasets[@]}"; do
    echo "$password" | sudo -S  $(which python) EEG_level_head.py \
            --mode predict \
            --dataset ${FOC_GEN_SPIKES_dataset} \
            --task_model checkpoints/${FOC_GEN_SPIKES_dataset}_EEGlevel.pth \
            --test_csv_dir ${result_dir}/pred_FOCGENSPIKES_1sStep \
            --result_dir ${result_dir}
done


# Spike
  # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/SPIKES.pth \
            --abs_pos_emb \
            --dataset SPIKES \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SPIKES_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}


  # (2). EEG_level prediction
echo "$password" | sudo -S  $(which python) EEG_level_head.py \
          --mode predict \
          --dataset SPIKES \
          --task_model checkpoints/SPIKES_EEGlevel.pth \
          --test_csv_dir ${result_dir}/pred_SPIKES_1sStep  \
          --result_dir ${result_dir} \
          --align_spike_detection_and_location


# IIIC
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/IIIC.pth \
            --abs_pos_emb \
            --dataset IIIC \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_IIIC_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}


    # (2). EEG_level prediction (SEIZURE, LPD, GPD, LRDA, GRDA are seperated)
IIIC_datasets=("SEIZURE" "LPD" "GPD" "LRDA" "GRDA")
for IIIC_dataset in "${IIIC_datasets[@]}"; do
    echo "$password" | sudo -S  $(which python) EEG_level_head.py \
            --mode predict \
            --dataset ${IIIC_dataset} \
            --task_model checkpoints/${IIIC_dataset}_EEGlevel.pth \
            --test_csv_dir ${result_dir}/pred_IIIC_1sStep \
            --result_dir ${result_dir}
done


# Sleep 5 stage
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/SLEEPPSG.pth \
            --abs_pos_emb \
            --dataset SLEEPPSG \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLEEPPSG_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}

echo password | sudo -S $(which python)  EEG_level_head.py \
        --mode predict \
        --dataset SLEEPPSG \
        --test_csv_dir  ${result_dir}/pred_SLEEPPSG_1sStep \
        --result_dir  ${result_dir}


# Sleep 3 stage
    # (1). Continuous 1-second step event-level prediction
echo "$password" | sudo OMP_NUM_THREADS=1 $(which python) finetune_classification.py \
            --predict \
            --model morgoth_backbone_base \
            --task_model checkpoints/SLEEP.pth \
            --abs_pos_emb \
            --dataset MGBSLEEP3stages \
            --data_format ${data_format} \
            --sampling_rate ${sampling_rate} \
            --already_format_channel_order ${already_format_channel_order} \
            --already_average_montage ${already_average_montage} \
            --allow_missing_channels ${allow_missing_channels} \
            --leave_one_hemisphere_out ${leave_one_hemisphere_out} \
            --polarity ${polarity} \
            --eval_sub_dir ${dataset_dir} \
            --eval_results_dir ${result_dir}/pred_SLEEP3stages_1sStep \
            --prediction_slipping_step_second 1 \
            --device cpu \
            --distributed False \
            --rewrite_results ${rewrite_results}


echo "$password" | sudo -S $(which python)  EEG_level_head.py \
        --mode predict \
        --dataset SLEEP3stages \
        --test_csv_dir  ${result_dir}/pred_SLEEP3stages_1sStep \
        --result_dir  ${result_dir}





