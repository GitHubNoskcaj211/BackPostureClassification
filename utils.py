from matplotlib import pyplot as plt
import numpy as np
import json
import torch
import os

mps_device = torch.device("mps")
dataset_path = 'Dataset/Cleaned Dataset/'

def plot_uwb_data_tensor(tensor):
    fig, axes = plt.subplots(3, 6)
    tspan = np.arange(tensor.shape[0])
    
    def plot_axis(axis, data_sequence, title):
        axis.set_title(title)
        axis.plot(tspan, data_sequence)

    plot_axis(axes[0,0], tensor[:,0], 'Device 0 | Distance')
    plot_axis(axes[0,1], tensor[:,1], 'Device 0 | Dir Vector[0]')
    plot_axis(axes[0,2], tensor[:,2], 'Device 0 | Dir Vector[1]')
    plot_axis(axes[0,3], tensor[:,3], 'Device 0 | Dir Vector[2]')
    plot_axis(axes[0,4], tensor[:,4], 'Device 0 | Elevation')
    plot_axis(axes[0,5], tensor[:,5], 'Device 0 | Azimuth')
    plot_axis(axes[1,0], tensor[:,6], 'Device 1 | Distance')
    plot_axis(axes[1,1], tensor[:,7], 'Device 1 | Dir Vector[0]')
    plot_axis(axes[1,2], tensor[:,8], 'Device 1 | Dir Vector[1]')
    plot_axis(axes[1,3], tensor[:,9], 'Device 1 | Dir Vector[2]')
    plot_axis(axes[1,4], tensor[:,10], 'Device 1 | Elevation')
    plot_axis(axes[1,5], tensor[:,11], 'Device 1 | Azimuth')
    plot_axis(axes[2,0], tensor[:,12], 'Device 2 | Distance')
    plot_axis(axes[2,1], tensor[:,13], 'Device 2 | Dir Vector[0]')
    plot_axis(axes[2,2], tensor[:,14], 'Device 2 | Dir Vector[1]')
    plot_axis(axes[2,3], tensor[:,15], 'Device 2 | Dir Vector[2]')
    plot_axis(axes[2,4], tensor[:,16], 'Device 2 | Elevation')
    plot_axis(axes[2,5], tensor[:,17], 'Device 2 | Azimuth')

    fig.show()

def plot_metrics_over_epochs(metrics_file_path):
    with open(metrics_file_path, 'r') as metrics_file:
        lines = metrics_file.readlines()
        x = np.empty(len(lines))
        metric_keys = json.loads(lines[0].replace("\'", "\"").split(' | ')[1].split(':: ')[-1]).keys()
        training_metrics = np.empty((len(lines), len(metric_keys)))
        testing_metrics = np.empty((len(lines), len(metric_keys)))
        for ii, line in enumerate(lines):
            line = line.replace("\'", "\"")
            segments = [segment.split(':: ')[-1] for segment in line.split(' | ')]
            x[ii] = int(segments[0])
            training_dict = json.loads(segments[1])
            for jj, value in enumerate(training_dict.values()):
                training_metrics[ii,jj] = float(value)
            testing_dict = json.loads(segments[2])
            for jj, value in enumerate(testing_dict.values()):
                testing_metrics[ii,jj] = float(value)

        def plot_axis(axis, training_sequence, testing_sequence, title):
            axis.set_title(title)
            axis.plot(x, training_sequence, c='blue', label='Training')
            axis.plot(x, testing_sequence, c='orange', label='Testing')
                        

        fig, axes = plt.subplots(len(metric_keys), 1)
        for jj, metric_key in enumerate(metric_keys):
            plot_axis(axes[jj], training_metrics[:,jj], testing_metrics[:,jj], f'Metric: {metric_key}')
        axes[0].legend()
        fig.suptitle    (f'Metrics for {metrics_file_path}')
        fig.show()

def compute_k_fold_cross_validation_metrics(directory):
    training_metric_values = {'Loss': [], 'Good Posture Accuracy': [], 'Bad Posture Accuracy': []}
    testing_metric_values = {'Loss': [], 'Good Posture Accuracy': [], 'Bad Posture Accuracy': []}
    for subdirectory in [x[0] for x in os.walk(directory)][1:]:
        with open(f'{subdirectory}/metrics.txt', 'r') as metrics_file:
            lines = metrics_file.readlines()
            epoch_chosen_line = lines[-1]
            epoch_chosen_line = epoch_chosen_line.replace("\'", "\"")
            segments = [segment.split(':: ')[-1] for segment in epoch_chosen_line.split(' | ')]
            training_dict = json.loads(segments[1])
            for key, value in training_dict.items():
                training_metric_values[key].append(float(value))
            testing_dict = json.loads(segments[2])
            for key, value in testing_dict.items():
                testing_metric_values[key].append(float(value))

    for metric, values in training_metric_values.items():
        arr = np.array(values)
        print(f'Training {metric} average: {np.average(arr)}')
        print(f'Training {metric} std: {np.std(arr)}')
    for metric, values in testing_metric_values.items():
        arr = np.array(values)
        print(f'Testing {metric} average: {np.average(arr)}')
        print(f'Testing {metric} std: {np.std(arr)}')