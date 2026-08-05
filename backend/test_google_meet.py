"""
Test script for Google Meet Service
Tests if Google Calendar API is properly configured

USAGE:
    python test_google_meet.py

REQUIREMENTS:
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

from google_meet_service import google_meet_service
from datetime import datetime, timedelta

def test_google_meet_service():
    print("\n" + "="*60)
    print("Google Meet Service Test")
    print("="*60 + "\n")
    
    # Check if service is initialized
    print("1. Checking Google Calendar API initialization...")
    if google_meet_service.initialized:
        print("   ✅ Google Calendar API is properly configured")
        print("   ✅ Service account authenticated successfully")
    else:
        print("   ❌ Google Calendar API not configured")
        print("   ℹ️  Will use Jitsi Meet as fallback")
        print("   ℹ️  See GOOGLE_MEET_SETUP_GUIDE.md for setup instructions")
    
    print("\n2. Testing Meet Link Generation...")
    
    # Test appointment data
    test_appointment = {
        'appointmentId': f'TEST{int(datetime.now().timestamp())}',
        'appointmentDate': datetime.now() + timedelta(days=1),
        'timeSlot': '10-11',
        'doctorName': 'Dr. Test Smith',
        'patientName': 'John Test Doe',
        'doctorEmail': 'doctor@test.com',
        'patientEmail': 'patient@test.com'
    }
    
    print(f"   Creating meet link for test appointment...")
    print(f"   Appointment ID: {test_appointment['appointmentId']}")
    print(f"   Date: {test_appointment['appointmentDate'].strftime('%Y-%m-%d')}")
    print(f"   Time: {test_appointment['timeSlot']}")
    
    # Generate meet link
    meet_link = google_meet_service.create_meet_link(test_appointment)
    
    print(f"\n3. Result:")
    print(f"   Meet Link: {meet_link}")
    
    if 'meet.google.com' in meet_link:
        print("   ✅ REAL Google Meet link generated!")
        print("   ✅ Google Calendar API is working correctly")
        print(f"\n   Test the link: {meet_link}")
        print("   (Open in browser to verify it works)")
    elif 'meet.jit.si' in meet_link:
        print("   ⚠️  Fallback Jitsi Meet link generated")
        print("   ℹ️  This means Google Calendar API is not configured")
        print("   ℹ️  But Jitsi links work immediately - try opening it!")
        print(f"\n   Test the link: {meet_link}")
        print("   (Should open Jitsi room - works without setup)")
    else:
        print("   ❌ Unexpected link format")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60 + "\n")
    
    # Provide next steps
    if google_meet_service.initialized:
        print("✅ Your Google Meet integration is working!")
        print("\nNext steps:")
        print("1. Update app.py to include the meet link API endpoint")
        print("2. Update frontend to call backend for meet links")
        print("3. Test creating a real appointment")
    else:
        print("⚠️  Google Meet not configured - using Jitsi fallback")
        print("\nTwo options:")
        print("\nOption A: Use Jitsi Meet (Easy)")
        print("  ✓ Already working - no setup needed!")
        print("  ✓ Jitsi links work immediately")
        print("  ✓ Free and open-source")
        print("  → Just test creating an appointment - it should work!")
        print("\nOption B: Set up Google Meet (Advanced)")
        print("  1. Follow steps in GOOGLE_MEET_SETUP_GUIDE.md")
        print("  2. Create google-service-account.json")
        print("  3. Place it in backend/ directory")
        print("  4. Run this test script again")

if __name__ == '__main__':
    try:
        test_google_meet_service()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\nMake sure you have installed required packages:")
        print("pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
