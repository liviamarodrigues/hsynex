import os
import glob
import torch
import argparse
from create_input import CreateInput
from inference import InVivoInference

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--out_path", type=str)
parser.add_argument("--device", type=str, default = device)
parser.add_argument("--voxres", type=str, default = 'original')
args = parser.parse_args()
input_path_str = args.input_path
out_path_str = args.out_path

if os.path.isdir(input_path_str):
    list_input = sorted(glob.glob(os.path.join(input_path_str, '*')))
elif os.path.isfile(input_path_str):
    list_input = [input_path_str]
else:
    raise TypeError('input_path is not a file or directory')

for idx, path in enumerate(list_input):
    print('working on:', path)
    find_input = CreateInput()
    input_img, dilated_ss, vdc = find_input.extract_image(path, out_path_str)
    print('create_input_done')
    inference = InVivoInference(device = args.device , voxres = args.voxres)
    inference.predict(path, out_path_str, input_img, dilated_ss, vdc)