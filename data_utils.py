import torch
import numpy as np
from utils import dataset_path, mps_device
import json
import cv2
from tqdm import tqdm
from skimage.measure import block_reduce

sensor_ids = [15576551, 235261048, 185745774]
def convert_uwb_json_to_tensor(file_number):
    file_base = dataset_path + f'file{file_number:03}'
    file_path = file_base + '.log'
    with open(file_path) as file:
        json_file = json.load(file)
        starting_time = int(json_file[0]['timestamp'])
        ending_time = int(json_file[-1]['timestamp'])

        tensor = torch.empty((ending_time - starting_time, 12), dtype=torch.float32) # [distance, direction vector] per sensor
        sum_per_sensor = np.zeros((3, 4))
        num_hits_per_sensor = [0, 0, 0]
        current_time = starting_time
        for datum in json_file:
            datum_time = int(datum['timestamp'])
            if datum_time != current_time:
                averaging = np.eye(3)
                averaging[0, 0] = 1 / num_hits_per_sensor[0] if num_hits_per_sensor[0] != 0 else 1
                averaging[1, 1] = 1 / num_hits_per_sensor[1] if num_hits_per_sensor[1] != 0 else 1
                averaging[2, 2] = 1 / num_hits_per_sensor[2] if num_hits_per_sensor[2] != 0 else 1
                average_per_sensor = averaging @ sum_per_sensor
                average_tensor = torch.from_numpy(average_per_sensor)
                tensor[current_time - starting_time] = average_tensor.reshape(-1)
                sum_per_sensor = np.zeros((3, 4))
                num_hits_per_sensor = [0, 0, 0]
                current_time = datum_time

            sensor_index = sensor_ids.index(datum['deviceId'])
            num_hits_per_sensor[sensor_index] += 1
            sum_per_sensor[sensor_index, 0] += datum['distance']
            sum_per_sensor[sensor_index, 1] += datum['direction'][0]
            sum_per_sensor[sensor_index, 2] += datum['direction'][1]
            sum_per_sensor[sensor_index, 3] += datum['direction'][2]
        torch.save(tensor, f'{file_base}.tensor')

# Camera 1 (computer): 1620 x 1080 | 30 fps
# Camera 2 (phone): 1920 x 1080 | 29.95 fps
# TODO Fix the framerates (though it is only going to be off by a maximum of 12 frames which is close enough)... the data cleaning is likely more off than this
def convert_camera_to_tensor(file_number):
    computer_path = f'{dataset_path}Computer_file{file_number:03}.mov'
    phone_path = f'{dataset_path}Phone_file{file_number:03}.mov'
    computer_video = cv2.VideoCapture(computer_path)
    phone_video = cv2.VideoCapture(phone_path)
    computer_length = int(computer_video.get(cv2.CAP_PROP_FRAME_COUNT))
    phone_length = int(phone_video.get(cv2.CAP_PROP_FRAME_COUNT))
    data_length = min(phone_length, computer_length) - 30
    tensor = torch.empty((data_length, 6, 54, 96), dtype=torch.int8, device=mps_device)
    for ii in tqdm(range(data_length)):
        computer_ret, computer_frame = computer_video.read()
        phone_ret, phone_frame = phone_video.read()
        if not computer_ret:
            raise Exception('Unreachable.')
        if not phone_ret:
            raise Exception('Unreachable.')
        computer_frame = np.transpose(computer_frame, [2, 0, 1])
        new_computer_frame = np.zeros((3, 1080, 1920), dtype=np.int8)
        new_computer_frame[:,:,:1620] = computer_frame[:,:,:]
        computer_frame = new_computer_frame
        phone_frame = np.transpose(phone_frame, [2, 0, 1])
        reduced_computer_frame = arr_reduced = block_reduce(computer_frame, block_size=20, func=np.mean, cval=np.mean(computer_frame))
        reduced_phone_frame = arr_reduced = block_reduce(phone_frame, block_size=20, func=np.mean, cval=np.mean(phone_frame))
        tensor[ii,:3] = torch.from_numpy(reduced_computer_frame, device=mps_device)
        tensor[ii,3:] = torch.from_numpy(reduced_phone_frame, device=mps_device)
    
    torch.save(tensor, f'{dataset_path}video{file_number:03}.tensor')
    phone_video.release()
    computer_video.release()

for ii in range(1, 41):
    convert_camera_to_tensor(ii)

bad_posture_tensor = torch.tensor([0, 1], dtype=torch.float32, device=mps_device)
good_posture_tensor = torch.tensor([1, 0], dtype=torch.float32, device=mps_device)
file_number_to_label_map = {1: good_posture_tensor, 2: good_posture_tensor, 3: good_posture_tensor, 4: bad_posture_tensor, 5: good_posture_tensor, 6: bad_posture_tensor, 7: good_posture_tensor, 8: bad_posture_tensor, 9: bad_posture_tensor,
                     10: bad_posture_tensor, 11: good_posture_tensor, 12: bad_posture_tensor, 13: good_posture_tensor, 14: bad_posture_tensor, 15: good_posture_tensor, 16: bad_posture_tensor, 17: good_posture_tensor, 18: bad_posture_tensor, 19: good_posture_tensor,
                     20: bad_posture_tensor, 21: good_posture_tensor, 22: bad_posture_tensor, 23: good_posture_tensor, 24: bad_posture_tensor, 25: good_posture_tensor, 26: bad_posture_tensor, 27: good_posture_tensor, 28: bad_posture_tensor, 29: good_posture_tensor,
                     30: bad_posture_tensor, 31: good_posture_tensor, 32: bad_posture_tensor, 33: good_posture_tensor, 34: bad_posture_tensor, 35: good_posture_tensor, 36: bad_posture_tensor, 37: good_posture_tensor, 38: bad_posture_tensor, 39: good_posture_tensor,
                     40: bad_posture_tensor}

def get_data_uwb_train_test_split(train_file_numbers, test_file_numbers):
    assert not bool(set(train_file_numbers) & set(test_file_numbers)), 'Testing on data that is also in training set!'
    training_data = []
    for file_number in train_file_numbers:
        train_tensor = torch.load(f'{dataset_path}file{file_number:03}.tensor').to(mps_device)
        train_label = file_number_to_label_map[file_number]
        for ii in range(train_tensor.shape[0]):
            training_data.append((train_tensor[ii], train_label))

    testing_data = []
    for file_number in test_file_numbers:
        test_tensor = torch.load(f'{dataset_path}file{file_number:03}.tensor').to(mps_device)
        test_label = file_number_to_label_map[file_number]
        for ii in range(test_tensor.shape[0]):
            testing_data.append((test_tensor[ii], test_label))
    
    return training_data, testing_data

def get_data_camera_train_test_split(train_file_numbers, test_file_numbers):
    # assert not bool(set(train_file_numbers) & set(test_file_numbers)), 'Testing on data that is also in training set!'
    training_data = []
    for file_number in train_file_numbers:
        train_tensor = torch.load(f'{dataset_path}video{file_number:03}.tensor').to(mps_device).to(torch.float32) / 255.0
        train_label = file_number_to_label_map[file_number]
        for ii in range(train_tensor.shape[0]):
            training_data.append((train_tensor[ii], train_label))

    testing_data = []
    for file_number in test_file_numbers:
        test_tensor = torch.load(f'{dataset_path}video{file_number:03}.tensor').to(mps_device).to(torch.float32) / 255.0
        test_label = file_number_to_label_map[file_number]
        for ii in range(test_tensor.shape[0]):
            testing_data.append((test_tensor[ii], test_label))
    
    return training_data, testing_data