"""
Test script to verify the improved brain tumor model is loaded correctly in app.py
"""

import requests
import json

# Test the API
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("="*60)
    print("Testing Health Endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/api/health")
    data = response.json()
    print(json.dumps(data, indent=2))
    print()

def test_models_info():
    """Test models info endpoint"""
    print("="*60)
    print("Testing Models Info Endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/api/models")
    data = response.json()
    print(json.dumps(data, indent=2))
    print()

def test_brain_model_info():
    """Test brain model detailed info"""
    print("="*60)
    print("Testing Brain Model Info Endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/api/models/brain/info")
    data = response.json()
    print(json.dumps(data, indent=2))
    
    # Check version
    if 'version' in data:
        if data['version'] == 'v2_improved':
            print("\n✓ SUCCESS: Using IMPROVED v2 model!")
            print(f"  Test Accuracy: {data.get('test_accuracy', 'N/A')}")
        else:
            print("\n⚠️  WARNING: Using LEGACY v1 model")
            print("  Please train the improved model first!")
    print()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BRAIN TUMOR MODEL VERIFICATION TEST")
    print("="*60)
    print("\nMake sure Flask app is running on http://localhost:5000\n")
    
    try:
        test_health()
        test_models_info()
        test_brain_model_info()
        
        print("="*60)
        print("TEST COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to Flask app")
        print("Please start the Flask app first:")
        print("  cd backend")
        print("  python app.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")
