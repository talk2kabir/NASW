# NASGW
[GUI Widgets Classification based on Neural Architecture Search](#)
 
 This repository contains the description of the NASW approach as well as the dataset and trained models. 

# Supporting Documents
Paper [Journal](#)

Documents [Supporting Files](#)

Trained Models [GoogleDrive](https://drive.google.com/file/d/1tVqJ_buFrI_r2tKYY6LMvWdxGPjmmvra/view?usp=sharing)

Original Dataset [Zenodo](https://zenodo.org/record/2530277#.YWgqL0lfiUl)

Standard Dataset [Dropbox](https://www.dropbox.com/sh/dqy52o900ijyxz2/AAAslJzQ2slZqpQzI7ZRI6tia?dl=0)

# Approach 


![Fig. 6](https://github.com/talk2kabir/NASGW/blob/main/NASW_overview.png)

# Architecture Search & Testing

### One-shot Search
`python one_shot_search.py --tmp_data_dir ./search_data --save log_path` 

### Train Best Architecture
`python train_arch.py  --tmp_data_dir ./train_data --save log_path --auxiliary --note note_of_this_run` 

### Test Model
`python test_model.py` 

## Citation


