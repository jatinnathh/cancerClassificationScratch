"""
Direct startup script for VS Code terminal
Automatically uses torch_gpu environment
"""
import subprocess
import sys
import os

def find_conda_path():
    """Find conda installation"""
    import os
    possible_paths = [
        os.path.expandvars(r"C:\ProgramData\anaconda3"),
        os.path.expandvars(r"C:\Users\%USERNAME%\anaconda3"),
        os.path.expandvars(r"C:\Users\%USERNAME%\miniconda3"),
        os.path.expandvars(r"%USERPROFILE%\anaconda3"),
        os.path.expandvars(r"%USERPROFILE%\miniconda3"),
    ]
    
    for path in possible_paths:
        conda_exe = os.path.join(path, "Scripts", "conda.exe")
        if os.path.exists(conda_exe):
            return path
    return None

def get_torch_gpu_python():
    """Get python path from torch_gpu environment"""
    conda_path = find_conda_path()
    
    if not conda_path:
        print("❌ Conda not found. Please install Anaconda or Miniconda.")
        return None
    
    # Try to find torch_gpu environment
    torch_gpu_python = os.path.join(conda_path, "envs", "torch_gpu", "python.exe")
    
    if os.path.exists(torch_gpu_python):
        return torch_gpu_python
    
    print(f"❌ torch_gpu environment not found at: {torch_gpu_python}")
    print("\nPlease create it in Anaconda Prompt:")
    print("  conda create -n torch_gpu python=3.11 -y")
    print("  conda activate torch_gpu")
    print("  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y")
    print("  pip install flask flask-cors pillow opencv-python scikit-image werkzeug")
    return None

if __name__ == "__main__":
    print("="*60)
    print("Cancer Classification API - GPU Accelerated")
    print("="*60)
    print()
    
    # Get torch_gpu python
    torch_gpu_python = get_torch_gpu_python()
    
    if not torch_gpu_python:
        sys.exit(1)
    
    print(f"✓ Using: {torch_gpu_python}")
    print()
    
    # Run app.py with torch_gpu python
    print("Starting Flask server...")
    print()
    
    try:
        subprocess.run([torch_gpu_python, "app.py"])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
