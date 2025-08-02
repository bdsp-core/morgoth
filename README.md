
# 🧠 Morgoth: Toward Automated EEG Interpretation

This repository contains code and tools for running EEG analysis using **Morgoth**, including and event-level and EEG-level detection with probabilities for:

- Normal / Abnormal
- Slowing: No Slowing / Focal Slowing / Generalized Slowing
- Burst suppression: No / Burst suppression
- Spike detection: No / Spike
- Spike localization: No / Focal Spike / Generalized Spike  
- IIIC classification: Other / Seizure / LPD / GPD / LRDA / GRDA
- Sleep staging: Awake / N1 / N2 / N3 / REM
## 📁 Directory Structure

```
├── test_data/                  # EEG data (.mat/.pkl/.edf)
├── checkpoints/                # Pretrained models and checkpoints
├── xxx.py                      # Code files
├── xxx.sh/bat                  # Run model
└── README.md                   # Project overview
```
## 📥 Download Model and Test Data

Before running the code, please download the pretrained model (checkpoints) and test dataset (test_data) from Dropbox and place them in the appropriate folders:

- [Download Link – Model and Data](https://www.dropbox.com/scl/fo/6sb9kjeqcf0qr9ul399bt/AMBXz3vgkMrxS38tNyjapjc?rlkey=386p1uphrmewggutb8oup3pb5&st=kx1szipb&dl=0) 

## ⚙️ Setup

### 1. Create environment (conda recommended) in morgoth folder
### python 3.12 + pytorch 2.4 + CUDA 12.4

```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements.txt / conda install --file requirements.txt

# If you have GPU with cuda driver
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```


### 2. Optional: python 3.11 + pytorch 2.1 + CUDA 12.2 / 12.1
```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements_cuda122.txt

# If you have GPU with cuda driver
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. If Windows and no GPU
```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements_windows.txt
```


## 🚀 Usage

### Before Start

The model supports raw EEG data in both EDF and MAT formats as input.

This means you can use unaltered clinical recordings directly — no manual preprocessing is required. The model automatically performs all necessary preprocessing steps, including: 

Bandpass filtering, Resampling, Montage, Clipping, Normalization, and Epoching.

⚠️ To enable this pipeline, users must either:

- Ensure the raw EEG file contains sampling rate and channel names, or

- Provide this information explicitly via command-line arguments, or 

- Channel order must follow the standard as specified in the corresponding bash scripts. 

Please refer to the continuous_event_level.sh, discrete_event_level.sh, or EEG_level.sh for example configurations and expected input parameters.

This design ensures a streamlined and reproducible workflow, allowing you to run the model on raw EEG files directly, without requiring prior signal processing expertise.

### 1. Run continuous prediction on event level

See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

```bash
bash continuous_event_level.sh
```

### 2. Run discrete prediction
See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

The data in test_data folder should be processed first.

```bash
bash discrete_event_level.sh
```

### 3. Run EEG level prediction
See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

The EEG_level inputs are the event-level output results with 1-second slipping step.

```bash
bash EEG_level.sh
```

### 4.Optional: If linux and cpu

```bash
bash bash EEG_level_cpu.sh
```

### 5.Optional: If windows and cpu

```bash
EEG_level_windows_cpu.bat
```

## 🏋️‍♂️ Train a Model from Scratch

To train a Morgoth model from scratch using your own data:

### 1. Prepare your dataset in `.h5` for pretraining and `.pkl` for fine tuning 

You should modify the data_provider.py script according to your dataset

```bash
echo password | sudo -S ~/miniconda3/envs/torchenv/bin/python data_provider.py
```

### 2. Run the pretraining script:

```bash
bash pretrain.sh
```

### 3. Run the fine-tuning script for event-level:

```bash
bash train_classification.sh
```

### 4. Run the fine-tuning script for EEG-level:

```bash
bash train_EEG_level_head.sh
```

You may modify the script or config file to set the number of epochs, learning rate, batch size, model type, etc.

Make sure you have sufficient GPU memory for large models or long EEG recordings.


## 🔧 Errors and Solutions

### NumPy and Pandas Binary Incompatibility

Reinstall NumPy using Conda with the command:

```bash
conda install numpy=1.26.4
```


## 🚨 Update 

### 2025-06-08

Add layer clip for cuda index out of boundary in EEG_level_head.sh, class CNNTransformerClassifier, def forward

```bash
# check the valid of index
if indices.max() >= x.size(1) or indices.min() < 0:
    indices = torch.clamp(indices, 0, x.size(1) - 1)
```

### 2025-06-10

When encountering pandas and numpy compatibility issues, please reinstall NumPy 1.26.4 using Conda instead of pip.

```bash
conda install numpy=1.26.4
```

### 2025-07-24

#### Running Morgoth on Long EEGs by Segmenting

If you cannnot run Morgoth on long EEG files because of GPU memory limitations. To perform continuous event-level detection, run the following command:

```bash

bash continuous_event_level_longeeg.sh

```
This script will automatically segment the EEG files and batch them through Morgoth. The final output will contain predictions for each segment. In the script, you’ll need to specify the following arguments:

```bash

--data_format edf \                               # format of the raw EEG data 
--segment_duration 600 \                          # duration of each segment in seconds (here, 10 minutes) 
--eeg_dir test_data/longEEG/raw \                 # directory containing the original raw EEG files 
--eval_sub_dir test_data/longEEG/segments_raw \   # directory to store the segmented files

```

continuous_event_level_longeeg.sh included examples in the script for IIIC, spike, slowing, and burst suppression detection. You can follow a similar format for other tasks. Just make sure to keep the --eval_sub_dir path consistent across all commands.

If you need to merge the segment-level results into a full-length prediction, note that you can’t just concatenate them — due to the sliding step, you’ll need to align each segment by padding the predictions before merging.


### 2025-08-02

Wrong code in utils.py resize_array_along_axis0()

Change 

```bash
return arr[indices:,]
```

to 

```bash
return arr[indices,: ]
```


## 📬 Contact

For questions, please contact:  
Chenxi Sun – csun8@bidmc.harvard.edu
