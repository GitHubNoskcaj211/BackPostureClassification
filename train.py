import torch
import math
import os
from tqdm import tqdm
from data_utils import bad_posture_tensor, good_posture_tensor
from utils import mps_device
import torchvision
import copy

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
        

def train_model(model, optimizer, loss_fn, save_directory, num_epochs, num_models_to_save, num_batches, training_data, testing_data, transformations):
       
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
            if transformations != None:
                input = transformations(input)
            predictions = model(input)
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), f'{save_directory}model{epoch}.pth')
    metrics_file.close()

def k_fold_cross_validation_uwb(loss_fn, save_directory, num_epochs, num_models_to_save, num_batches, folds, get_train_test_split, transformations):
    data_per_fold = [get_train_test_split(fold[0], fold[1]) for fold in folds]
    for fold_number, (training_data_per_fold, testing_data_per_fold) in enumerate(data_per_fold):
        print(f'Fold {fold_number} / {len(folds)}')
        model = UWBCategoricalPredictor(18, [36, 36], 2)
        model.to(mps_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
        train_model(model, optimizer, loss_fn, f'{save_directory}fold{fold_number}/', num_epochs, num_models_to_save, num_batches, training_data_per_fold, testing_data_per_fold, transformations)

def k_fold_cross_validation_camera(loss_fn, save_directory, num_epochs, num_models_to_save, num_batches, folds, get_train_test_split, transformations):
    data_per_fold = [get_train_test_split(fold[0], fold[1]) for fold in folds]
    c = 0
    for fold_number, (training_data_per_fold, testing_data_per_fold) in enumerate(data_per_fold):
        if c < 4:
            c += 1
            continue
        print(f'Fold {fold_number} / {len(folds)}')
        model = ImageCategoricalPredictor((54, 96), 2)
        model.to(mps_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
        train_model(model, optimizer, loss_fn, f'{save_directory}fold{fold_number}/', num_epochs, num_models_to_save, num_batches, training_data_per_fold, testing_data_per_fold, transformations)

from models import UWBCategoricalPredictor
from data_utils import get_data_uwb_train_test_split
from utils import plot_metrics_over_epochs, compute_k_fold_cross_validation_metrics

task1_folds = [([2, 3, 4, 5, 6, 7, 8, 9],[1, 10]), ([1, 3, 4, 5, 6, 7, 8, 10],[2, 9]), ([1, 2, 4, 5, 6, 7, 9, 10],[3, 8]), ([1, 2, 3, 5, 6, 8, 9, 10],[4, 7]), ([1, 2, 3, 4, 7, 8, 9, 10],[5, 6])]
task2_folds = [([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],[11, 12]), 
               ([11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],[13, 14]), 
               ([11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],[15, 16]), 
               ([11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26],[17, 18]), 
               ([11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26],[19, 20]), 
               ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26],[21, 22]), 
               ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26],[23, 24]), 
               ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],[25, 26])]
task3_folds = [([29, 30, 31, 32, 33, 34, 35, 36],[27, 28]),
               ([27, 28, 31, 32, 33, 34, 35, 36],[29, 30]),
               ([27, 28, 29, 30, 33, 34, 35, 36],[31, 32]),
               ([27, 28, 29, 30, 31, 32, 35, 36],[33, 34]),
               ([27, 28, 29, 30, 31, 32, 33, 34],[35, 36])]
task4_folds = [([37, 38],[39, 40]), ([39, 40],[37, 38])]

# UWB
def uwb_transform(tensor):
    new_tensor = tensor.detach().clone().to(mps_device)
    new_tensor[:,[0,6,12]] = tensor[:,[0,6,12]] + (torch.rand((tensor.shape[0],3), device=mps_device) - 0.5) / 10
    new_tensor[:,[1,2,3,7,8,9,13,14,15]] = tensor[:,[1,2,3,7,8,9,13,14,15]] + (torch.rand((tensor.shape[0],9), device=mps_device) - 0.5) / 5
    new_tensor[:,[4,5,10,11,16,17]] = tensor[:,[4,5,10,11,16,17]] + (torch.rand((tensor.shape[0],6), device=mps_device) - 0.5) * 10
    return new_tensor
loss_fn = torch.nn.BCELoss()

# UWB Hyperparameter Tuning
# uwb_training_data_task_1, uwb_testing_data_task_1 = get_data_uwb_train_test_split([1, 2, 3, 4, 6, 8, 5, 9],[7, 10])
# uwb_training_data_task_2, uwb_testing_data_task_2 = get_data_uwb_train_test_split([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],[25, 26])
# uwb_training_data_task_3, uwb_testing_data_task_3 = get_data_uwb_train_test_split([27, 28, 29, 30, 31, 32, 33, 34],[35, 36])
# uwb_training_data_task_4, uwb_testing_data_task_4 = get_data_uwb_train_test_split([37, 38],[39, 40])
# model = UWBCategoricalPredictor(18, [36, 36], 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/uwb_test_task1/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_1, testing_data=uwb_testing_data_task_1, transformations=uwb_transform)
# model = UWBCategoricalPredictor(18, [36, 36], 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/uwb_test_task2/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_1, testing_data=uwb_testing_data_task_1, transformations=uwb_transform)
# model = UWBCategoricalPredictor(18, [36, 36], 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/uwb_test_task3/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_1, testing_data=uwb_testing_data_task_1, transformations=uwb_transform)
# model = UWBCategoricalPredictor(18, [36, 36], 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/uwb_test_task4/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=uwb_training_data_task_1, testing_data=uwb_testing_data_task_1, transformations=uwb_transform)
# plot_metrics_over_epochs('Runs/uwb_test_task1/metrics.txt')
# plot_metrics_over_epochs('Runs/uwb_test_task2/metrics.txt')
# plot_metrics_over_epochs('Runs/uwb_test_task3/metrics.txt')
# plot_metrics_over_epochs('Runs/uwb_test_task4/metrics.txt')
# input()

# UWB k-fold cross
# k_fold_cross_validation_uwb(loss_fn, save_directory='Runs/kfold_uwb_task1/', num_epochs=100, num_models_to_save=0, num_batches=10, folds=task1_folds, get_train_test_split=get_data_uwb_train_test_split, transformations=uwb_transform)
# k_fold_cross_validation_uwb(loss_fn, save_directory='Runs/kfold_uwb_task2/', num_epochs=100, num_models_to_save=0, num_batches=10, folds=task2_folds, get_train_test_split=get_data_uwb_train_test_split, transformations=uwb_transform)
# k_fold_cross_validation_uwb(loss_fn, save_directory='Runs/kfold_uwb_task3/', num_epochs=100, num_models_to_save=0, num_batches=10, folds=task3_folds, get_train_test_split=get_data_uwb_train_test_split, transformations=uwb_transform)
# k_fold_cross_validation_uwb( loss_fn, save_directory='Runs/kfold_uwb_task4/', num_epochs=100, num_models_to_save=0, num_batches=10, folds=task4_folds, get_train_test_split=get_data_uwb_train_test_split, transformations=uwb_transform)
# print('Task1')
# compute_k_fold_cross_validation_metrics('Runs/kfold_uwb_task1/')
# print('Task2')
# compute_k_fold_cross_validation_metrics('Runs/kfold_uwb_task2/')
# print('Task3')
# compute_k_fold_cross_validation_metrics('Runs/kfold_uwb_task3/')
# print('Task4')
# compute_k_fold_cross_validation_metrics('Runs/kfold_uwb_task4/')


# Camera
from models import ImageCategoricalPredictor
from data_utils import get_data_camera_train_test_split
from utils import plot_metrics_over_epochs

def camera_transform(tensor):
    transforms = torch.nn.Sequential(
        torchvision.transforms.ColorJitter(.1, .1, .1, .1),
        torchvision.transforms.GaussianBlur((5, 5)),
        torchvision.transforms.RandomResizedCrop(size=(54, 96), scale=(0.5, 1), ratio=(1, 1), antialias=None),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    )
    new_tensor = tensor.detach().clone().to(mps_device)
    new_tensor[:,:3] = transforms(tensor[:,:3])
    new_tensor[:,3:] = transforms(tensor[:,3:])
    return new_tensor
loss_fn = torch.nn.BCELoss()

# Camera hyperparameter tuning
# camera_training_data_task_1, camera_testing_data_task_1 = get_data_camera_train_test_split([1, 2, 3, 4, 6, 8, 5, 9],[7, 10])
# camera_training_data_task_2, camera_testing_data_task_2 = get_data_camera_train_test_split([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],[25, 26])
# camera_training_data_task_3, camera_testing_data_task_3 = get_data_camera_train_test_split([27, 28, 29, 30, 31, 32, 33, 34],[35, 36])
# camera_training_data_task_4, camera_testing_data_task_4 = get_data_camera_train_test_split([37, 38],[39, 40])
# model = ImageCategoricalPredictor((54, 96), 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/camera_test_task1/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=camera_training_data_task_1, testing_data=camera_testing_data_task_1, transformations=camera_transform)
# model = ImageCategoricalPredictor((54, 96), 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/camera_test_task2/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=camera_training_data_task_2, testing_data=camera_testing_data_task_2, transformations=camera_transform)
# model = ImageCategoricalPredictor((54, 96), 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/camera_test_task3/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=camera_training_data_task_3, testing_data=camera_testing_data_task_3, transformations=camera_transform)
# model = ImageCategoricalPredictor((54, 96), 2)
# model.to(mps_device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)
# train_model(model, optimizer, loss_fn, save_directory='Runs/camera_test_task4/', num_epochs=100, num_models_to_save=0, num_batches=10, training_data=camera_training_data_task_4, testing_data=camera_testing_data_task_4, transformations=camera_transform)
# plot_metrics_over_epochs('Runs/camera_test_task1/metrics.txt')
# plot_metrics_over_epochs('Runs/camera_test_task2/metrics.txt')
# plot_metrics_over_epochs('Runs/camera_test_task3/metrics.txt')
# plot_metrics_over_epochs('Runs/camera_test_task4/metrics.txt')
# input()

# Camera k-fold cross
# k_fold_cross_validation_camera(loss_fn, save_directory='Runs/kfold_camera_task1/', num_epochs=80, num_models_to_save=0, num_batches=10, folds=task1_folds, get_train_test_split=get_data_camera_train_test_split, transformations=camera_transform)
k_fold_cross_validation_camera(loss_fn, save_directory='Runs/kfold_camera_task2/', num_epochs=60, num_models_to_save=0, num_batches=10, folds=task2_folds, get_train_test_split=get_data_camera_train_test_split, transformations=camera_transform)
# k_fold_cross_validation_camera(loss_fn, save_directory='Runs/kfold_camera_task3/', num_epochs=60, num_models_to_save=0, num_batches=10, folds=task3_folds, get_train_test_split=get_data_camera_train_test_split, transformations=camera_transform)
# k_fold_cross_validation_camera(loss_fn, save_directory='Runs/kfold_camera_task4/', num_epochs=80, num_models_to_save=0, num_batches=10, folds=task4_folds, get_train_test_split=get_data_camera_train_test_split, transformations=camera_transform)
print('Task1')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task1/')
print('Task2')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task2/')
print('Task3')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task3/')
print('Task4')
compute_k_fold_cross_validation_metrics('Runs/kfold_camera_task4/')