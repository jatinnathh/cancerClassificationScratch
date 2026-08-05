"""
GPU Information and Test Script
Checks if CUDA is available and displays GPU details
"""

import torch
import sys

print("="*70)
print("GPU CONFIGURATION CHECK")
print("="*70)

# Check CUDA availability
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n2. CUDA Version: {torch.version.cuda}")
    print(f"3. PyTorch Version: {torch.__version__}")
    print(f"4. Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i} Details:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - CUDA Capability: {props.major}.{props.minor}")
        print(f"   - Multi Processors: {props.multi_processor_count}")
    
    # Test tensor creation on GPU
    print(f"\n5. Testing GPU Tensor Creation...")
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"   ✓ Successfully created tensor on GPU")
        print(f"   Tensor device: {test_tensor.device}")
        
        # Test computation
        result = test_tensor @ test_tensor.T
        print(f"   ✓ Successfully performed matrix multiplication on GPU")
        
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Memory info
    print(f"\n6. GPU Memory Status:")
    print(f"   - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"   - Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    print(f"\n{'='*70}")
    print("✓ GPU is ready for classification!")
    print("="*70)
    
else:
    print(f"\n✗ CUDA is NOT available")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"\nPossible reasons:")
    print("  1. No NVIDIA GPU detected")
    print("  2. CUDA drivers not installed")
    print("  3. PyTorch installed without CUDA support")
    print(f"\nTo install PyTorch with CUDA support, run:")
    print("  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print(f"\n{'='*70}")
    print("⚠ Will run on CPU (slower performance)")
    print("="*70)
    sys.exit(1)
