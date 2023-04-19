from utils import plot_uwb_data_tensor, dataset_path, plot_metrics_over_epochs, compute_k_fold_cross_validation_metrics
import torch
from data_utils import convert_uwb_json_to_tensor

# plot_uwb_data_tensor(torch.load(f'{dataset_path}file004.tensor'))
# input()

# plot_metrics_over_epochs('Runs/kfold_camera_task2/fold2/metrics.txt')
# input()

print('Task1')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task1/')
print('Task2')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task2/')
print('Task3')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task3/')
print('Task4')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task4/')