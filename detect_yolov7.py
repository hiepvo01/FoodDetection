import os
import subprocess
import torch

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

def detect_objects(image_path, weights='runs/train/exp/weights/best.pt', device='0'):
    cmd = [
        "python", "detect.py",
        "--weights", weights,
        "--conf", "0.25",
        "--img-size", "640",
        "--source", image_path,
        "--device", device
    ]
    return run_command(cmd)

if __name__ == "__main__":
    cuda_available = check_cuda()
    device = "0" if cuda_available else "cpu"

    # Make sure we're in the yolov7 directory
    if not os.path.exists("yolov7"):
        print("YOLOv7 directory not found. Please run the training script first.")
        exit(1)
    os.chdir("yolov7")

    custom_image_path = "../Food-ingredient-recognition-4/test/images/10_jpg.rf.d2e67b2a31285a9747e0567cfcf5ec5a.jpg"
    weights_path = "runs/train/exp/weights/best.pt"

    print(f"Running object detection on {custom_image_path}")
    if not detect_objects(custom_image_path, weights=weights_path, device=device):
        print("Object detection failed. Please check the error messages above.")
    else:
        print("Object detection completed successfully.")
        print(f"Results can be found in the 'runs/detect' directory")

# Example usage:
# python detect_yolov7.py