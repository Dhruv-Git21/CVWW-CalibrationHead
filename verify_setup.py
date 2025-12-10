"""
Setup verification script - checks if all dependencies are installed correctly.

Usage:
    python verify_setup.py
"""
import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (PyTorch sees {torch.cuda.device_count()} GPU(s))")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA is NOT available (training will use CPU - slower)")
            return False
    except:
        return False

def check_file_structure():
    """Check if all required files exist."""
    required_files = [
        "requirements.txt",
        "README.md",
        "configs/cifar100_resnet50.yaml",
        "configs/cifar100_wrn2810.yaml",
        "configs/cifar100_vit_tiny.yaml",
        "configs/detector_retinanet_voc.yaml",
        "src/train_cls.py",
        "src/eval_cls.py",
        "src/train_det.py",
        "src/eval_det.py",
        "src/datasets/cifar100.py",
        "src/datasets/voc.py",
        "src/models/resnet.py",
        "src/models/vit.py",
        "src/models/retinanet.py",
        "src/utils/common.py",
        "src/utils/train_utils.py",
        "src/utils/detector_utils.py",
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            pass  # Don't print each file to avoid clutter
        else:
            print(f"✗ Missing file: {file}")
            all_exist = False
    
    if all_exist:
        print("✓ All required files exist")
    
    return all_exist

def main():
    print("=" * 60)
    print("Setup Verification for CV_calibration Project")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info >= (3, 10):
        print("✓ Python version is 3.10 or higher")
    else:
        print("⚠ Python 3.10+ is recommended (current: {}.{})".format(
            sys.version_info.major, sys.version_info.minor))
    print()
    
    # Check file structure
    print("Checking file structure...")
    check_file_structure()
    print()
    
    # Check core dependencies
    print("Checking core dependencies...")
    all_installed = True
    
    # PyTorch ecosystem
    all_installed &= check_import("torch", "PyTorch")
    all_installed &= check_import("torchvision")
    all_installed &= check_import("timm")
    
    # Data science
    all_installed &= check_import("numpy")
    all_installed &= check_import("sklearn", "scikit-learn")
    all_installed &= check_import("matplotlib")
    all_installed &= check_import("seaborn")
    
    # Utilities
    all_installed &= check_import("yaml", "PyYAML")
    all_installed &= check_import("tqdm")
    all_installed &= check_import("cv2", "opencv-python")
    all_installed &= check_import("PIL", "Pillow")
    
    # TensorBoard
    all_installed &= check_import("tensorboard")
    
    # Optional
    print("\nChecking optional dependencies...")
    check_import("pycocotools")
    
    print()
    
    # Check CUDA
    print("Checking GPU support...")
    check_cuda()
    print()
    
    # Summary
    print("=" * 60)
    if all_installed:
        print("✓ Setup verification PASSED!")
        print("\nYou're ready to start training!")
        print("\nQuick start:")
        print("  python src/train_cls.py --config configs/cifar100_resnet50.yaml")
        print("\nFor more info, see README.md and QUICKSTART.md")
    else:
        print("✗ Setup verification FAILED")
        print("\nSome dependencies are missing. Install them with:")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == '__main__':
    main()
