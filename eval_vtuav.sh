python tracking/analysis_results_vtuav.py --dataset_name VTUAV --tracker_param baseline | tee eval_vtuav.log
python tracking/analysis_results_vtuav.py --dataset_name VTUAV --dataset_split _short --tracker_param baseline | tee -a eval_vtuav.log
python tracking/analysis_results_vtuav.py --dataset_name VTUAV --dataset_split _long --tracker_param baseline | tee -a eval_vtuav.log