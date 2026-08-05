# 🎯 Quick Reference - GPU Backend

## 🚀 Quick Start Commands

### First Time Setup:
```bash
conda create -n torch_gpu python=3.11 -y
conda activate torch_gpu
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install flask flask-cors pillow opencv-python scikit-image werkzeug
```

### Every Time You Start:
```bash
cd backend
start_gpu.bat
```

## ✅ Verification

**Check GPU:**
```bash
python check_gpu.py
```

**Check from Python:**
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🎛️ Startup Scripts

| Script | Purpose | Use When |
|--------|---------|----------|
| `start_gpu.bat` | Quick start with torch_gpu | Regular use |
| `start_server.bat` | Full startup with checks | First time / debugging |
| `python app.py` | Manual start | Advanced users |

## 📊 Expected Output

### ✅ Success (GPU Enabled):
```
============================================================
Device Configuration:
============================================================
Using device: cuda
GPU Name: NVIDIA GeForce RTX 3060
CUDA Version: 11.8
GPU Memory: 12.00 GB
GPU Available: ✓ ENABLED
============================================================
```

### ⚠️ Fallback (CPU Only):
```
============================================================
Using device: cpu
GPU Available: ✗ Running on CPU (slower)
============================================================
```

## 🔧 Common Commands

### Monitor GPU:
```bash
nvidia-smi -l 1
```

### Clear GPU Memory:
```python
import torch
torch.cuda.empty_cache()
```

### Switch Environment:
```bash
conda deactivate
conda activate torch_gpu
```

### Check PyTorch CUDA:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| CUDA not available | `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia` |
| Environment not found | `conda create -n torch_gpu python=3.11` |
| Out of memory | Close other GPU apps, restart server |
| Slow first run | Normal - models loading into GPU |

## ⚡ Performance

- **With GPU**: 0.3-1 second per image
- **Without GPU**: 2-5 seconds per image
- **Speedup**: 3-10x faster!

## 📞 Need Help?

1. Check `GPU_SETUP_GUIDE.md` for detailed instructions
2. Run `python check_gpu.py` to diagnose issues
3. Check server logs for error messages

---

**Environment**: torch_gpu (Conda)  
**Device**: CUDA (GPU) or CPU fallback  
**Framework**: PyTorch with CUDA support
