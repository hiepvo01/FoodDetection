import os
import yaml
from roboflow import Roboflow
import torch
import subprocess
import logging
from tqdm import tqdm
​
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
​
def check_cuda():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"CUDA is available. CUDA version: {cuda_version}")
        logging.info(f"GPU device: {device_name}")
    else:
        logging.info("CUDA is not available. Using CPU.")
    return cuda_available
​
def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    if process.returncode != 0:
        logging.error(f"Error executing command: {' '.join(cmd)}")
        logging.error(f"Error message: {error}")
        return False
    return True
​
def download_dataset():
    rf = Roboflow(api_key="ksHd7Xl6jWDdulCaiRNs")
    project = rf.workspace("thanh-huy-phan").project("food-ingredient-recognition")
    version = project.version(4)
    dataset = version.download("yolov7")
    logging.info(f"Dataset downloaded to: {os.path.abspath(dataset.location)}")
    return dataset
​
def setup_yolov7():
    if not os.path.exists("yolov7"):
        subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov7"])
    os.chdir("yolov7")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    if not os.path.exists("yolov7.pt"):
        subprocess.run(["wget", "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"])
    logging.info(f"YOLOv7 setup complete in: {os.path.abspath(os.getcwd())}")
​
def prepare_dataset_config(dataset):
    with open(os.path.join(dataset.location, 'data.yaml'), 'r') as file:
        data = yaml.safe_load(file)
    data['train'] = os.path.join(dataset.location, 'train', 'images')
    data['val'] = os.path.join(dataset.location, 'valid', 'images')
    data['test'] = os.path.join(dataset.location, 'test', 'images')
    config_path = 'data/custom_data.yaml'
    with open(config_path, 'w') as file:
        yaml.dump(data, file)
    logging.info(f"Custom dataset configuration saved to: {os.path.abspath(config_path)}")
​
def train(epochs, batch_size, weights='yolov7.pt', device='0'):
    cmd = [
        "python", "train.py",
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", "data/custom_data.yaml",
        "--weights", weights,
        "--device", device,
        "--nosave",  # Don't save checkpoints to reduce disk usage
        "--cache"    # Cache images for faster training
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    # Set up progress bar
    pbar = tqdm(total=epochs, desc="Training Progress")
    
    current_epoch = 0
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # Print the line in real-time
        if "Epoch" in line and "/" in line:
            try:
                epoch_info = line.split()[2]  # Extract epoch information
                epoch_num = int(epoch_info.split('/')[0])  # Extract current epoch number
                if epoch_num > current_epoch:
                    current_epoch = epoch_num
                    pbar.update(1)  # Update progress bar
                    logging.info(f"Completed epoch {current_epoch}/{epochs}")
            except ValueError:
                pass  # In case of unexpected format, just continue
    
    pbar.close()
    
    # Check for any errors
    error = process.stderr.read()
    if error:
        logging.error(f"Error during training: {error}")
        return False
    
    return True
​
def test(weights='runs/train/exp/weights/best.pt', device='0'):
    if not os.path.exists(weights):
        logging.warning(f"Weights file not found: {weights}")
        exp_dir = 'runs/train'
        weight_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(exp_dir) for f in filenames if f.endswith('.pt')]
        if weight_files:
            latest_weights = max(weight_files, key=os.path.getmtime)
            weights = latest_weights
            logging.info(f"Using latest weights file: {weights}")
        else:
            logging.error("No weights file found. Please check the training output.")
            return False
​
    cmd = [
        "python", "test.py",
        "--data", "data/custom_data.yaml",
        "--weights", weights,
        "--device", device
    ]
    return run_command(cmd)
​
if __name__ == "__main__":
    try:
        cuda_available = check_cuda()
        device = "0" if cuda_available else "cpu"
        
        dataset = download_dataset()
        setup_yolov7()
        prepare_dataset_config(dataset)
        
        epochs = 100
        batch_size = 16
        
        logging.info(f"Starting training with device: {device}")
        if not train(epochs, batch_size, device=device):
            logging.error("Training failed. Please check the error messages above.")
        else:
            logging.info("Training completed successfully.")
        
        logging.info(f"Starting testing with device: {device}")
        if not test(device=device):
            logging.error("Testing failed. Please check the error messages above.")
        else:
            logging.info("Testing completed successfully.")
    
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
