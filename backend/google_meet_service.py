"""
Google Meet Service for Flask
Creates real Google Meet links using Google Calendar API

SETUP REQUIRED:
1. Go to: https://console.cloud.google.com
2. Create a new project or select existing
3. Enable Google Calendar API
4. Create Service Account credentials
5. Download JSON key file
6. Save as: backend/google-service-account.json
7. Install: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import os
import json

class GoogleMeetService:
    def __init__(self):
        self.service = None
        self.initialized = False
        self.initialize()
    
    def initialize(self):
        """Initialize Google Calendar API with service account"""
        try:
            # Path to service account key file
            key_path = os.path.join(os.path.dirname(__file__), '../../google-service-account.json')
            
            if not os.path.exists(key_path):
                print(f"⚠️  Google service account key not found at: {key_path}")
                print(f"⚠️  Using fallback Jitsi Meet links. Set up Google Calendar API for real Meet links.")
                self.initialized = False
                return False
            
            # Load credentials
            credentials = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=['https://www.googleapis.com/auth/calendar']
            )
            
            # Build Calendar API service
            self.service = build('calendar', 'v3', credentials=credentials)
            self.initialized = True
            
            print("✅ Google Calendar API initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Google Calendar API: {str(e)}")
            self.initialized = False
            return False
    
    def create_meet_link(self, appointment_data):
        """
        Create a Google Meet link for an appointment
        
        Args:
            appointment_data (dict): {
                'appointmentId': str,
                'appointmentDate': datetime or timestamp,
                'timeSlot': str (e.g., '10-11'),
                'doctorName': str,
                'patientName': str,
                'doctorEmail': str (optional),
                'patientEmail': str (optional)
            }
        
        Returns:
            str: Google Meet link or fallback Jitsi link
        """
        # If not initialized, return fallback link
        if not self.initialized:
            return self._generate_fallback_link(appointment_data['appointmentId'])
        
        try:
            appointment_id = appointment_data['appointmentId']
            time_slot = appointment_data['timeSlot']
            doctor_name = appointment_data['doctorName']
            patient_name = appointment_data['patientName']
            
            # Parse time slot (e.g., "10-11" means 10:00 AM to 11:00 AM)
            start_hour, end_hour = map(int, time_slot.split('-'))
            
            # Get appointment date
            appointment_date = appointment_data['appointmentDate']
            if isinstance(appointment_date, str):
                appointment_date = datetime.fromisoformat(appointment_date)
            elif hasattr(appointment_date, 'toDate'):  # Firebase Timestamp
                appointment_date = appointment_date.toDate()
            
            # Create start and end datetime
            start_time = appointment_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            end_time = appointment_date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            
            # Create calendar event with Google Meet
            event = {
                'summary': f'Medical Consultation: Dr. {doctor_name} & {patient_name}',
                'description': f'Telemedicine appointment\nAppointment ID: {appointment_id}\nDoctor: {doctor_name}\nPatient: {patient_name}',
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'Asia/Kolkata',  # Adjust timezone as needed
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'Asia/Kolkata',
                },
                'conferenceData': {
                    'createRequest': {
                        'requestId': appointment_id,
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
                'attendees': []
            }
            
            # Add attendees if emails provided
            if 'doctorEmail' in appointment_data and appointment_data['doctorEmail']:
                event['attendees'].append({'email': appointment_data['doctorEmail']})
            if 'patientEmail' in appointment_data and appointment_data['patientEmail']:
                event['attendees'].append({'email': appointment_data['patientEmail']})
            
            # Insert event
            created_event = self.service.events().insert(
                calendarId='primary',
                conferenceDataVersion=1,
                body=event
            ).execute()
            
            # Extract Google Meet link
            meet_link = None
            if 'conferenceData' in created_event and 'entryPoints' in created_event['conferenceData']:
                for entry in created_event['conferenceData']['entryPoints']:
                    if entry['entryPointType'] == 'video':
                        meet_link = entry['uri']
                        break
            
            if meet_link:
                print(f"✅ Created Google Meet link: {meet_link}")
                return meet_link
            else:
                print("⚠️  No Meet link in response, using fallback")
                return self._generate_fallback_link(appointment_id)
        
        except HttpError as error:
            print(f"❌ Google Calendar API Error: {error}")
            return self._generate_fallback_link(appointment_data['appointmentId'])
        except Exception as e:
            print(f"❌ Error creating Google Meet link: {str(e)}")
            return self._generate_fallback_link(appointment_data['appointmentId'])
    
    def _generate_fallback_link(self, appointment_id):
        """Generate fallback Jitsi Meet link when Google API is unavailable"""
        import time
        timestamp = int(time.time() * 1000)
        meet_link = f"https://meet.jit.si/ongogenesis-{appointment_id}-{timestamp}"
        print(f"📎 Using fallback Jitsi Meet link: {meet_link}")
        return meet_link
    
    def delete_meet_link(self, appointment_id, event_id=None):
        """
        Delete/Cancel a Google Meet event
        
        Args:
            appointment_id (str): Appointment ID
            event_id (str): Calendar event ID (if known)
        
        Returns:
            bool: Success status
        """
        if not self.initialized:
            return True  # Nothing to delete for fallback links
        
        try:
            # If we have the event ID, delete it
            if event_id:
                self.service.events().delete(
                    calendarId='primary',
                    eventId=event_id
                ).execute()
                print(f"🗑️  Deleted Google Meet event: {event_id}")
            else:
                print(f"🗑️  Logging meet link deletion for appointment: {appointment_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting meet link: {str(e)}")
            return False

# Create singleton instance
google_meet_service = GoogleMeetService()
