import gdown

import os
import argparse

parser = argparse.ArgumentParser(description='Download data and checkpoints')
parser.add_argument('--data', action='store_true', help='Download data')
parser.add_argument('--checkpoints', action='store_true', help='Download checkpoints')
parser.add_argument('--data_dir', type=str, default='./Data', help='Directory to save data. Default: ./Data')
parser.add_argument('--checkpoint_dir', type=str, default='./Checkpoints', help='Directory to save checkpoints. Default: ./Checkpoints')

args = parser.parse_args()

checkpoint_ids = {
    '64x64': '13y4UdoxBMwZnO-3Oz3dWCVlgfkwqwLd1',
    '256x256': '1VAnEhX1GTLYQXOTq-f1kZperQJTTbYyC',
    '64x64_256x256': '1JX1M9EOrpWfwEUUwFXrNGeEyObFk9gYP',
    'All': '18ocK4a9zV2v5Zv986z_VdYC0AK-QzZtn'
}

data_ids = {
    'topologies.npy': '1VitnaTfJtkEqY5jFIdfyB132-8Gu7s7u',
    'boundary_conditions.npy': '1CYqL9BMR6PiM9PfE81VZIWyPIK1hDzAR',
    'shapes.npy': '1g6A152bwJEQwh0Bvr9xQUHeBboT4gRPT',
    'loads.npy': '1ZMvQk7J_kKaAaTpr2BYkm8B4E-6E3hRB',
    'vfs.npy': '1WFeDNY_qwWeVSCVoSqIh_SrPXmf8Xsym'
}

test_data_ids = {
    'topologies.npy': '1tFa2twksRnhc67XR47yQ-aeWnwZFfVe7',
    'boundary_conditions.npy': '1tXzECt2Gb2__lXVG799jc1hvWMTqWu2T',
    'shapes.npy': '1tFa2twksRnhc67XR47yQ-aeWnwZFfVe7',
    'loads.npy': '126ysGYKE9RynNM814QNzQK94knhEtm03',
    'vfs.npy': '1HlKcqpZ78gjVolEFUrwQVJW1vQRdOS88'
}

# check if directories exist
if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

if not os.path.exists(os.path.join(args.data_dir, 'Test')):
    os.makedirs(os.path.join(args.data_dir, 'Test'))

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if args.data:
    for file in data_ids.keys():
        gdown.download(id=data_ids[file], output=os.path.join(args.data_dir, file))
    
    for file in test_data_ids.keys():
        gdown.download(id=test_data_ids[file], output=os.path.join(args.data_dir, 'Test', file))

if args.checkpoints:
    for file in checkpoint_ids.keys():
        gdown.download(id=checkpoint_ids[file], output=os.path.join(args.checkpoint_dir, file, 'checkpoint_epoch_50.pth'))
    