@echo off
REM Windows Batch Script for EEG analysis
REM Save this file with .bat extension

REM Set environment variables
set OMP_NUM_THREADS=1

REM Define variables
set dataset_dir=test_data\Sandor\EDF
set data_format=edf
set sampling_rate=0
set result_dir=test_data\Sandor\results
set already_format_channel_order=no
set already_average_montage=no
set allow_missing_channels=no
set leave_one_hemisphere_out=no
set polarity=-1
set rewrite_results=no

REM Create results directory if it doesn't exist
if not exist %result_dir% mkdir %result_dir%

echo Starting EEG analysis processes...

REM Normal: (1) Continuous 1-second step event-level prediction
echo Running NORMAL event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\NORMAL.pth ^
    --abs_pos_emb ^
    --dataset NORMAL ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_NORMAL_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM Normal: (2) EEG_level prediction
echo Running NORMAL EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset NORMAL ^
    --task_model checkpoints\NORMAL_EEGlevel.pth ^
    --test_csv_dir test_data\Sandor\pred_NORMAL_1sStep ^
    --result_dir test_data\Sandor ^
    --device cpu

REM Slowing: (1) Continuous 1-second step event-level prediction
echo Running SLOWING event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\SLOWING.pth ^
    --abs_pos_emb ^
    --dataset SLOWING ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_SLOWING_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM Slowing: (2) EEG_level prediction - FOC_SLOWING
echo Running FOC_SLOWING EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset FOC_SLOWING ^
    --task_model checkpoints\FOC_SLOWING_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_SLOWING_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM Slowing: (2) EEG_level prediction - GEN_SLOWING
echo Running GEN_SLOWING EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset GEN_SLOWING ^
    --task_model checkpoints\GEN_SLOWING_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_SLOWING_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM BS: (1) Continuous 1-second step event-level prediction
echo Running BS event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\BS.pth ^
    --abs_pos_emb ^
    --dataset BS ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_BS_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM BS: (2) EEG_level prediction
echo Running BS EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset BS ^
    --task_model checkpoints\BS_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_BS_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM FOC GEN SPIKES: (1) Continuous 1-second step event-level prediction
echo Running FOC GEN SPIKES event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\FOCGENSPIKES.pth ^
    --abs_pos_emb ^
    --dataset FOC_GEN_SPIKES ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_FOCGENSPIKES_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM FOC GEN SPIKES: (2) EEG_level prediction - FOC_SPIKES
echo Running FOC_SPIKES EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset FOC_SPIKES ^
    --task_model checkpoints\FOC_SPIKES_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_FOCGENSPIKES_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM FOC GEN SPIKES: (2) EEG_level prediction - GEN_SPIKES
echo Running GEN_SPIKES EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset GEN_SPIKES ^
    --task_model checkpoints\GEN_SPIKES_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_FOCGENSPIKES_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM Spike: (1) Continuous 1-second step event-level prediction
echo Running SPIKES event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\SPIKES.pth ^
    --abs_pos_emb ^
    --dataset SPIKES ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_SPIKES_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM Spike: (2) EEG_level prediction
echo Running SPIKES EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset SPIKES ^
    --task_model checkpoints\SPIKES_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_SPIKES_1sStep ^
    --result_dir %result_dir% ^
    --align_spike_detection_and_location ^
    --device cpu

REM IIIC: (1) Continuous 1-second step event-level prediction
echo Running IIIC event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\IIIC.pth ^
    --abs_pos_emb ^
    --dataset IIIC ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_IIIC_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM IIIC: (2) EEG_level prediction - SEIZURE
echo Running SEIZURE EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset SEIZURE ^
    --task_model checkpoints\SEIZURE_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_IIIC_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM IIIC: (2) EEG_level prediction - LPD
echo Running LPD EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset LPD ^
    --task_model checkpoints\LPD_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_IIIC_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM IIIC: (2) EEG_level prediction - GPD
echo Running GPD EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset GPD ^
    --task_model checkpoints\GPD_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_IIIC_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM IIIC: (2) EEG_level prediction - LRDA
echo Running LRDA EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset LRDA ^
    --task_model checkpoints\LRDA_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_IIIC_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM IIIC: (2) EEG_level prediction - GRDA
echo Running GRDA EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset GRDA ^
    --task_model checkpoints\GRDA_EEGlevel.pth ^
    --test_csv_dir %result_dir%\pred_IIIC_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM Sleep 5 stage: (1) Continuous 1-second step event-level prediction
echo Running SLEEPPSG event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\SLEEPPSG.pth ^
    --abs_pos_emb ^
    --dataset SLEEPPSG ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_SLEEPPSG_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM Sleep 5 stage: (2) EEG_level prediction
echo Running SLEEPPSG EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset SLEEPPSG ^
    --test_csv_dir %result_dir%\pred_SLEEPPSG_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

REM Sleep 3 stage: (1) Continuous 1-second step event-level prediction
echo Running SLEEP3stages event-level prediction...
python finetune_classification.py ^
    --predict ^
    --model morgoth_backbone_base ^
    --task_model checkpoints\SLEEP.pth ^
    --abs_pos_emb ^
    --dataset MGBSLEEP3stages ^
    --data_format %data_format% ^
    --sampling_rate %sampling_rate% ^
    --already_format_channel_order %already_format_channel_order% ^
    --already_average_montage %already_average_montage% ^
    --allow_missing_channels %allow_missing_channels% ^
    --leave_one_hemisphere_out %leave_one_hemisphere_out% ^
    --polarity %polarity% ^
    --eval_sub_dir %dataset_dir% ^
    --eval_results_dir %result_dir%\pred_SLEEP3stages_1sStep ^
    --prediction_slipping_step_second 1 ^
    --device cpu ^
    --distributed False ^
    --rewrite_results %rewrite_results%

REM Sleep 3 stage: (2) EEG_level prediction
echo Running SLEEP3stages EEG-level prediction...
python EEG_level_head.py ^
    --mode predict ^
    --dataset SLEEP3stages ^
    --test_csv_dir %result_dir%\pred_SLEEP3stages_1sStep ^
    --result_dir %result_dir% ^
    --device cpu

echo All EEG analysis processes completed!
pause