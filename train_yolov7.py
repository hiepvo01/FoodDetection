import os
import yaml
from roboflow import Roboflow
import torch
import subprocess

def check_cuda():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. CUDA version: {cuda_version}")
        print(f"GPU device: {device_name}")
    else:
        print("CUDA is not available. Using CPU.")
    return cuda_available

def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error message: {error}")
        return False
    return True

def download_dataset():
    rf = Roboflow(api_key="ksHd7Xl6jWDdulCaiRNs")
    project = rf.workspace("thanh-huy-phan").project("food-ingredient-recognition")
    version = project.version(4)
    dataset = version.download("yolov7")
    print(f"Dataset downloaded to: {os.path.abspath(dataset.location)}")
    return dataset

def setup_yolov7():
    if not os.path.exists("yolov7"):
        subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov7"])
    os.chdir("yolov7")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    if not os.path.exists("yolov7.pt"):
        subprocess.run(["wget", "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"])
    print(f"YOLOv7 setup complete in: {os.path.abspath(os.getcwd())}")

def prepare_dataset_config(dataset):
    with open(os.path.join(dataset.location, 'data.yaml'), 'r') as file:
        data = yaml.safe_load(file)

    data['train'] = os.path.join(dataset.location, 'train', 'images')
    data['val'] = os.path.join(dataset.location, 'valid', 'images')
    data['test'] = os.path.join(dataset.location, 'test', 'images')

    config_path = 'data/custom_data.yaml'
    with open(config_path, 'w') as file:
        yaml.dump(data, file)
    print(f"Custom dataset configuration saved to: {os.path.abspath(config_path)}")

def train(epochs, batch_size, weights='yolov7.pt', resume=False, device='0'):
    cmd = [
        "python", "train.py",
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", "data/custom_data.yaml",
        "--weights", 'runs/train/exp/weights/last.pt' if resume else weights,
        "--device", device
    ]
    if resume:
        cmd.append("--resume")
    return run_command(cmd)

def test(weights='runs/train/exp/weights/best.pt', device='0'):
    cmd = [
        "python", "test.py",
        "--data", "data/custom_data.yaml",
        "--weights", weights,
        "--device", device
    ]
    return run_command(cmd)

if __name__ == "__main__":
    cuda_available = check_cuda()
    device = "0" if cuda_available else "cpu"

    dataset = download_dataset()
    setup_yolov7()
    prepare_dataset_config(dataset)

    epochs = 100
    batch_size = 16

    print(f"Starting training with device: {device}")
    if not train(epochs, batch_size, device=device):
        print("Training failed. Please check the error messages above.")
    else:
        print("Training completed successfully.")

    print(f"Starting testing with device: {device}")
    if not test(device=device):
        print("Testing failed. Please check the error messages above.")
    else:
        print("Testing completed successfully.")

# Example usage:
# python train_yolov7.py