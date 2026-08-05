# 🚀 Run from VS Code Terminal

## Method 1: PowerShell Script (Recommended)
```powershell
.\start_gpu.ps1
```

## Method 2: Python Launcher
```powershell
python run_gpu.py
```

## Method 3: Direct Command (if you know conda path)
```powershell
C:\ProgramData\anaconda3\envs\torch_gpu\python.exe app.py
```
Or
```powershell
C:\Users\YOUR_USERNAME\anaconda3\envs\torch_gpu\python.exe app.py
```

## Method 4: Use Anaconda Prompt Integration
In VS Code terminal, type:
```powershell
conda activate torch_gpu
python app.py
```

---

## Quick Setup for VS Code

1. **Open Anaconda Prompt** (not VS Code terminal)
2. **Create environment:**
   ```
   conda create -n torch_gpu python=3.11 -y
   conda activate torch_gpu
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
   pip install flask flask-cors pillow opencv-python scikit-image werkzeug
   ```

3. **Return to VS Code terminal and run:**
   ```powershell
   .\start_gpu.ps1
   ```

---

## VS Code Terminal Configuration

To make conda work in VS Code terminal permanently:

1. Open VS Code Settings (Ctrl+,)
2. Search for "terminal integrated shell"
3. Add this to settings.json:
   ```json
   "terminal.integrated.profiles.windows": {
     "Anaconda": {
       "path": "C:\\Windows\\System32\\cmd.exe",
       "args": ["/K", "C:\\ProgramData\\anaconda3\\Scripts\\activate.bat"]
     }
   },
   "terminal.integrated.defaultProfile.windows": "Anaconda"
   ```

Then restart VS Code terminal.
