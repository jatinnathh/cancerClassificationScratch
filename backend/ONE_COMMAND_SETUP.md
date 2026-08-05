# ⚡ One-Command GPU Setup

Copy and paste these commands to set up GPU-accelerated backend:

## Step 1: Create Environment (First Time Only)

```bash
conda create -n torch_gpu python=3.11 -y && conda activate torch_gpu && conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && pip install flask flask-cors pillow opencv-python scikit-image werkzeug
```

## Step 2: Start Server (Every Time)

```bash
cd backend && conda activate torch_gpu && python app.py
```

---

## Or Use the Batch File:

```bash
cd backend
start_gpu.bat
```

---

## Verify GPU:

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

That's it! 🎉
