import torch
import math
import os
from tqdm import tqdm
from data_utils import bad_posture_tensor, good_posture_tensor
from utils import mps_device

@torch.no_grad()
def evalutate_model(model, input, labels, loss_fn):
    metrics = {}
    predictions = model(input)
    loss = loss_fn(predictions, labels)
    one_hot_predictions = torch.zeros(predictions.shape, device=mps_device)
    one_hot_predictions[(torch.arange(len(predictions)).unsqueeze(1), torch.topk(predictions,1).indices)] = 1
    num_good_posture = torch.sum(labels @ good_posture_tensor)
    num_bad_posture = torch.sum(labels @ bad_posture_tensor)
    num_correct_good_posture = torch.sum((one_hot_predictions * labels) @ good_posture_tensor)
    num_correct_bad_posture = torch.sum((one_hot_predictions * labels) @ bad_posture_tensor)
    good_posture_accuracy = num_correct_good_posture / num_good_posture
    bad_posture_accuracy = num_correct_bad_posture / num_bad_posture

    metrics['Loss'] = loss.item()
    metrics['Good Posture Accuracy'] = good_posture_accuracy.item()
    metrics['Bad Posture Accuracy'] = bad_posture_accuracy.item()
    
    return metrics
        

def train_model(model, optimizer, loss_fn, save_directory, num_epochs, num_models_to_save, num_batches, training_data, testing_data):
       
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=math.ceil(len(training_data) / num_batches), shuffle=True)
    full_training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=len(training_data))
    testing_data_loader = torch.utils.data.DataLoader(testing_data, batch_size=len(testing_data))
    
    number_epochs_between_model_saves = int(num_epochs / (num_models_to_save + 1)) + 1

    metrics_file_path = f'{save_directory}metrics.txt'
    os.makedirs(os.path.dirname(metrics_file_path))
    metrics_file = open(metrics_file_path, 'w')

    for epoch in tqdm(range(num_epochs)):
        testing_metrics = {}
        for testing_input, testing_label in testing_data_loader:
            assert len(testing_metrics) == 0
            testing_metrics = evalutate_model(model, testing_input, testing_label, loss_fn)
        training_metrics = {}
        for full_training_input, full_training_label in full_training_data_loader:
            assert len(training_metrics) == 0
            training_metrics = evalutate_model(model, full_training_input, full_training_label, loss_fn)
        metrics_file.write(f'Epoch:: {epoch} | Training:: {str(training_metrics)} | Testing:: {str(testing_metrics)}\n')

        if epoch % number_epochs_between_model_saves == 0 and epoch != 0:
            torch.save(model.state_dict(), f'{save_directory}model{epoch}.pth')

        for (input, labels) in training_data_loader:
            predictions = model(input)
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), f'{save_directory}model{epoch}.pth')
    metrics_file.close()

from models import UWBCategoricalPredictor
from data_utils import get_data_uwb_train_test_split
from utils import plot_metrics_over_epochs
# model = UWBCategoricalPredictor(12, [5], 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = torch.nn.BCELoss()
# uwb_training_data_task_1, uwb_testing_data_task_1 = get_data_uwb_train_test_split([1, 2, 3, 4, 6, 8, 5, 9],[7, 10])
# train_model(model, optimizer, loss_fn, save_directory='Runs/test1/', num_epochs=1000, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_1, testing_data=uwb_testing_data_task_1)
# plot_metrics_over_epochs('Runs/test1/metrics.txt')
# input()
# model = UWBCategoricalPredictor(12, [5], 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = torch.nn.BCELoss()
# uwb_training_data_task_2, uwb_testing_data_task_2 = get_data_uwb_train_test_split([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],[25, 26])
# train_model(model, optimizer, loss_fn, save_directory='Runs/test2/', num_epochs=1000, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_2, testing_data=uwb_testing_data_task_2)
# plot_metrics_over_epochs('Runs/test2/metrics.txt')
# input()
# model = UWBCategoricalPredictor(12, [2], 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = torch.nn.BCELoss()
# uwb_training_data_task_3, uwb_testing_data_task_3 = get_data_uwb_train_test_split([1, 2, 3, 4, 6, 8, 5, 9],[7, 10])
# train_model(model, optimizer, loss_fn, save_directory='Runs/test3/', num_epochs=1000, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_3, testing_data=uwb_testing_data_task_3)
# plot_metrics_over_epochs('Runs/test3/metrics.txt')
# input()
# model = UWBCategoricalPredictor(12, [2], 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = torch.nn.BCELoss()
# uwb_training_data_task_4, uwb_testing_data_task_4 = get_data_uwb_train_test_split([1, 2, 3, 4, 6, 8, 5, 9],[7, 10])
# train_model(model, optimizer, loss_fn, save_directory='Runs/test4/', num_epochs=1000, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_4, testing_data=uwb_testing_data_task_4)
# plot_metrics_over_epochs('Runs/test4/metrics.txt')
# input()

from models import ImageCategoricalPredictor
from data_utils import get_data_camera_train_test_split
from utils import plot_metrics_over_epochs
model = ImageCategoricalPredictor((54, 96), 2)
model.to(mps_device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
video_training_data_task_1, video_testing_data_task_1 = get_data_camera_train_test_split([1],[1])
train_model(model, optimizer, loss_fn, save_directory='Runs/video_test1/', num_epochs=1000, num_models_to_save=0, num_batches=10, training_data=video_training_data_task_1, testing_data=video_testing_data_task_1)
plot_metrics_over_epochs('Runs/video_test1/metrics.txt')
input()

# Find good training hyper parameters task 1
# k-fold cross validation
# Metrics average and std
# camera model
# find good training hyper parameters task 1
# k-fold cross validation
# repeat for 4 tasks