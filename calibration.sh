
# IIIC

class_number=(0 1 2 3 4 5)
class_name=('Other' 'Seizure' 'LPD' 'GPD' 'LRDA' 'GRDA')

for ((i=0; i<${#class_number[@]}; i++ )); do
  echo "exxact@1" | sudo -S  $(which python) calibration.py train \
      --file_path /home/exx/Desktop/calibration/IIIC/IIICOutput_Morgoth_SPaRCNet_KaggleWinner_30experts.csv \
      --pred_column M_class_${class_number[$i]}_prob \
      --class_number ${class_number[$i]} \
      --calibration_task_dir /home/exx/Desktop/calibration/IIIC/ \
      --expert_columns 'Aaron F. Struck' 'Aline Herlopian' 'Andres Rodriguez' \
      'Arcot Jayagopal Lakshman' 'Brian Appavu' 'Christa B. Swisher' \
      'Emily Gilmore' 'Emily L. Johnson' 'Eric S. Rosenthal' \
      'Gamaleldin Osman' 'Hiba A. Haider' 'Manisha Holmes ' 'Ioannis Karakis' \
      'Jay Pathmanathan' 'Jiyeoun Yoo' 'Jonathan J. Halford' \
      'Jong Woo Lee' 'M. Brandon Westover' 'Mackenzie C. Cervenka' \
      'Marcus Ng' 'Mohammad Tabaeizadeh F.' 'Monica B. Dhakar' \
      'Nicolas Gaspard' 'Peter W. Kaplan' 'Rani Sarkis' 'Sahar Zafar' \
      'Sarah Schmitt' 'Susan T. Herman' 'Olga Taraschenko' 'Zubeda Sheikh'


  echo "exxact@1" | sudo -S  $(which python) calibration.py apply \
  --new_data_path /home/exx/Desktop/calibration/IIIC/IIICOutput_Morgoth_SPaRCNet_KaggleWinner_30experts.csv \
  --pred_column  M_class_${class_number[$i]}_prob \
  --output_path /home/exx/Desktop/calibration/IIIC/results/${class_name[$i]}_calibration.csv \
  --class_name ${class_name[$i]} \
  --platt_model_path /home/exx/Desktop/calibration/IIIC/models/${class_number[$i]}_platt_model.pkl \
  --isotonic_model_path /home/exx/Desktop/calibration/IIIC/models/${class_number[$i]}_isotonic_model.pkl \
  --soft_score_column soft_label_${class_number[$i]}

done



#SN2
echo "exxact@1" | sudo -S  $(which python) calibration.py train \
    --file_path /home/exx/Desktop/calibration/SN2/SpikesOutput_Morgoth_SpikeNet_24experts.csv \
    --pred_column M_pred \
    --calibration_task_dir /home/exx/Desktop/calibration/SN2/ \


echo "exxact@1" | sudo -S  $(which python) calibration.py apply \
  --new_data_path /home/exx/Desktop/calibration/SN2/SpikesOutput_Morgoth_SpikeNet_24experts.csv \
  --pred_column  M_pred \
  --output_path /home/exx/Desktop/calibration/SN2/results/Spikes_calibration.csv \
  --class_name Spikes \
  --platt_model_path /home/exx/Desktop/calibration/SN2/models/platt_model.pkl \
  --isotonic_model_path /home/exx/Desktop/calibration/SN2/models/isotonic_model.pkl \
  --soft_score_column soft_score

#SAI
file_list=('NormalOutput_Morgoth_ScoreAI_experts.xlsx' 'FocalSlowingOutput_Morgoth_ScoreAI_experts.xlsx' 'GenSlowingOutput_Morgoth_ScoreAI_experts.xlsx' 'FocalSpikeOutput_Morgoth_ScoreAI_experts.xlsx' 'GenSpikeOutput_Morgoth_ScoreAI_experts.xlsx')
class_name=('Normal' 'FocalSlowing' 'GenSlowing' 'FocalSpike' 'GenSpike')

for ((i=0; i<${#file_list[@]}; i++ )); do
  echo "exxact@1" | sudo -S  $(which python) calibration.py train \
      --file_path /home/exx/Desktop/calibration/SAI/${file_list[$i]} \
      --pred_column M_pred \
      --class_name ${class_name[$i]} \
      --calibration_task_dir /home/exx/Desktop/calibration/SAI \
      --expert_columns 'expert_0' 'expert_1' 'expert_2' 'expert_3' 'expert_4' 'expert_5' 'expert_6' 'expert_8' 'expert_9' 'expert_10' 'expert_11' 'expert_12' 'expert_12' 'expert_14'

  echo "exxact@1" | sudo -S  $(which python) calibration.py apply \
    --new_data_path /home/exx/Desktop/calibration/SAI/${file_list[$i]}  \
    --pred_column  M_pred \
    --output_path /home/exx/Desktop/calibration/SAI/results/${class_name[$i]}_calibration.csv \
    --class_name ${class_name[$i]} \
    --platt_model_path /home/exx/Desktop/calibration/SAI/models/${class_name[$i]}_platt_model.pkl \
    --isotonic_model_path /home/exx/Desktop/calibration/SAI/models/${class_name[$i]}_isotonic_model.pkl \
    --soft_score_column soft_label
done


 UPenn
class_number=(0 1 2 3 4)
class_name=('Awake' 'N1' 'N2' 'N3' 'REM')

for ((i=0; i<${#class_number[@]}; i++ )); do
  echo "exxact@1" | sudo -S  $(which python) calibration.py train \
      --file_path /home/exx/Desktop/calibration/UPenn/UPennOutput_Morgoth_USleep_experts.csv \
      --pred_column M_class_${class_number[$i]}_prob \
      --class_number ${class_number[$i]} \
      --calibration_task_dir /home/exx/Desktop/calibration/UPenn/ \
      --expert_columns 'stage_expert_0' 'stage_expert_1' \
      'stage_expert_2' 'stage_expert_3' 'stage_expert_4' 'stage_expert_5'


  echo "exxact@1" | sudo -S  $(which python) calibration.py apply \
  --new_data_path /home/exx/Desktop/calibration/UPenn/UPennOutput_Morgoth_USleep_experts.csv \
  --pred_column  M_class_${class_number[$i]}_prob \
  --output_path /home/exx/Desktop/calibration/UPenn/results/${class_name[$i]}_calibration.csv \
  --class_name ${class_name[$i]} \
  --platt_model_path /home/exx/Desktop/calibration/UPenn/models/${class_number[$i]}_platt_model.pkl \
  --isotonic_model_path /home/exx/Desktop/calibration/UPenn/models/${class_number[$i]}_isotonic_model.pkl \
  --soft_score_column soft_label_${class_number[$i]}
done


# MoE
file_list=('SleepOutput_Morgoth_experts.csv' 'FocGenSlowingOutput_Morgoth_experts.csv' 'BurstSuppressionOutput_Morgoth_experts.csv' 'FocGenSpikeOutput_Morgoth_experts.csv' 'IIICOutput_Morgoth_experts.csv')

class_number_0=(0 1 2)
class_number_1=(1 2)
class_number_2=(1)
class_number_3=(1 2)
class_number_4=(1 2 3 4 5)

class_name_0=('Awake' 'N1' 'N2')
class_name_1=('FocalSlowing' 'GenSlowing')
class_name_2=('BurstSuppression')
class_name_3=('FocalSpike' 'GenSpike')
class_name_4=('Seizure' 'LPD' 'GPD' 'LRDA' 'GRDA')

for ((i=0; i<${#file_list[@]}; i++)); do
  class_number_var="class_number_$i[@]"
  class_name_var="class_name_$i[@]"
  class_numbers=("${!class_number_var}")
  class_names=("${!class_name_var}")

  for ((j=0; j<${#class_numbers[@]}; j++)); do
    class_num=${class_numbers[$j]}
    class_name=${class_names[$j]}

    echo "exxact@1" | sudo -S  $(which python) calibration.py train \
        --file_path  "/home/exx/Desktop/calibration/MoE/${file_list[$i]}" \
        --pred_column "class_${class_num}_prob" \
        --class_number "${class_num}" \
        --class_name "${class_name}" \
        --calibration_task_dir /home/exx/Desktop/calibration/MoE/ \
        --expert_columns 'expert_amorim' 'expert_bappavu' 'expert_bwestove' 'expert_danielgoldenholz' 'expert_ioanniskarakis' 'expert_jeroengijs' 'expert_jiyeounyoo' 'expert_jlee38' 'expert_larcotjayagopal' 'expert_marcusng' 'expert_olhataraschenko' 'expert_oselioutski' 'expert_osmangamaleldin' 'expert_pkaplan' 'expert_sfzafar' 'expert_ufisch' 'expert_wkong' 'expert_zubedasheikh' 'expert_afstruck' 'expert_humbertocastrolima' 'expert_alineherlopian'

    echo "exxact@1" | sudo -S  $(which python) calibration.py apply \
        --new_data_path "/home/exx/Desktop/calibration/MoE/${file_list[$i]}" \
        --pred_column "class_${class_num}_prob" \
        --output_path "/home/exx/Desktop/calibration/MoE/results/${class_name}_calibration.csv" \
        --class_name "${class_name}" \
        --platt_model_path "/home/exx/Desktop/calibration/MoE/models/${class_name}_platt_model.pkl" \
        --isotonic_model_path "/home/exx/Desktop/calibration/MoE/models/${class_name}_isotonic_model.pkl" \
        --soft_score_column "soft_label_${class_num}"
  done
done


#ON
file_list=('FocalSlowingOutput_Morgoth_experts.xlsx' 'GenSlowingOutput_Morgoth_experts.xlsx' 'FocalSpikeOutput_Morgoth_experts.xlsx' 'GenSpikeOutput_Morgoth_experts.xlsx')
class_name=('FocalSlowing' 'GenSlowing' 'FocalSpike' 'GenSpike')

for ((i=0; i<${#file_list[@]}; i++ )); do
  echo "exxact@1" | sudo -S  $(which python) calibration.py train \
      --file_path /home/exx/Desktop/calibration/ON/${file_list[$i]} \
      --pred_column M_pred \
      --class_name ${class_name[$i]} \
      --calibration_task_dir /home/exx/Desktop/calibration/ON \
      --expert_columns 'expert_20' 'expert_21' 'expert_24' 'expert_26' 'expert_27' 'expert_28' 'expert_30' 'expert_33' 'expert_36' 'expert_39' 'expert_40' 'expert_41' 'expert_46' 'expert_48'

  echo "exxact@1" | sudo -S  $(which python) calibration.py apply \
    --new_data_path /home/exx/Desktop/calibration/ON/${file_list[$i]}  \
    --pred_column M_pred \
    --output_path /home/exx/Desktop/calibration/ON/results/${class_name[$i]}_calibration.csv \
    --class_name ${class_name[$i]} \
    --platt_model_path /home/exx/Desktop/calibration/ON/models/${class_name[$i]}_platt_model.pkl \
    --isotonic_model_path /home/exx/Desktop/calibration/ON/models/${class_name[$i]}_isotonic_model.pkl \
    --isotonic_model_path /home/exx/Desktop/calibration/ON/models/${class_name[$i]}_isotonic_model.pkl \
    --soft_score_column soft_label
done