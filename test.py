from utils import plot_uwb_data_tensor, dataset_path
import torch

plot_uwb_data_tensor(torch.load(f'{dataset_path}file004.tensor'))
input()