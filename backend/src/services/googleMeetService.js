const { google } = require('googleapis');
const path = require('path');
const fs = require('fs');

/**
 * Google Meet Service
 * Creates real Google Meet links using Google Calendar API
 * 
 * SETUP REQUIRED:
 * 1. Go to: https://console.cloud.google.com
 * 2. Create a new project or select existing
 * 3. Enable Google Calendar API
 * 4. Create Service Account credentials
 * 5. Download JSON key file
 * 6. Save as: backend/google-service-account.json
 */

class GoogleMeetService {
  constructor() {
    this.calendar = null;
    this.initialized = false;
  }

  /**
   * Initialize Google Calendar API with service account
   */
  async initialize() {
    try {
      const keyFilePath = path.join(__dirname, '../../google-service-account.json');
      
      // Check if service account file exists
      if (!fs.existsSync(keyFilePath)) {
        console.warn('⚠️  Google service account key not found at:', keyFilePath);
        console.warn('⚠️  Using fallback mock links. Set up Google Calendar API for real Meet links.');
        this.initialized = false;
        return false;
      }

      const auth = new google.auth.GoogleAuth({
        keyFile: keyFilePath,
        scopes: ['https://www.googleapis.com/auth/calendar'],
      });

      const authClient = await auth.getClient();
      this.calendar = google.calendar({ version: 'v3', auth: authClient });
      this.initialized = true;
      
      console.log('✅ Google Calendar API initialized successfully');
      return true;
      
    } catch (error) {
      console.error('❌ Failed to initialize Google Calendar API:', error.message);
      this.initialized = false;
      return false;
    }
  }

  /**
   * Create a Google Meet link for an appointment
   */
  async createMeetLink(appointmentData) {
    // If not initialized, return fallback link
    if (!this.initialized) {
      return this.generateFallbackLink(appointmentData.appointmentId);
    }

    try {
      const { appointmentId, appointmentDate, timeSlot, doctorName, patientName } = appointmentData;

      // Parse time slot (e.g., "10-11" means 10:00 AM to 11:00 AM)
      const [startHour, endHour] = timeSlot.split('-').map(Number);
      
      // Create start and end datetime
      const appointmentDateObj = appointmentDate.toDate ? appointmentDate.toDate() : new Date(appointmentDate);
      const startDateTime = new Date(appointmentDateObj);
      startDateTime.setHours(startHour, 0, 0, 0);
      
      const endDateTime = new Date(appointmentDateObj);
      endDateTime.setHours(endHour, 0, 0, 0);

      // Create calendar event with Google Meet
      const event = {
        summary: `Medical Consultation: Dr. ${doctorName} & ${patientName}`,
        description: `Telemedicine appointment\nAppointment ID: ${appointmentId}\nDoctor: ${doctorName}\nPatient: ${patientName}`,
        start: {
          dateTime: startDateTime.toISOString(),
          timeZone: 'Asia/Kolkata', // Adjust timezone as needed
        },
        end: {
          dateTime: endDateTime.toISOString(),
          timeZone: 'Asia/Kolkata',
        },
        conferenceData: {
          createRequest: {
            requestId: appointmentId,
            conferenceSolutionKey: { type: 'hangoutsMeet' },
          },
        },
        attendees: [
          // Add doctor and patient emails if available
          // { email: doctorEmail },
          // { email: patientEmail },
        ],
      };

      const response = await this.calendar.events.insert({
        calendarId: 'primary',
        conferenceDataVersion: 1,
        requestBody: event,
      });

      // Extract Google Meet link
      const meetLink = response.data.conferenceData?.entryPoints?.[0]?.uri;
      
      if (meetLink) {
        console.log('✅ Created Google Meet link:', meetLink);
        return meetLink;
      } else {
        console.warn('⚠️  No Meet link in response, using fallback');
        return this.generateFallbackLink(appointmentId);
      }

    } catch (error) {
      console.error('❌ Error creating Google Meet link:', error.message);
      return this.generateFallbackLink(appointmentData.appointmentId);
    }
  }

  /**
   * Generate fallback Jitsi Meet link when Google API is unavailable
   */
  generateFallbackLink(appointmentId) {
    const timestamp = Date.now();
    const meetLink = `https://meet.jit.si/ongogenesis-${appointmentId}-${timestamp}`;
    console.log('📎 Using fallback Jitsi Meet link:', meetLink);
    return meetLink;
  }

  /**
   * Delete/Cancel a Google Meet event
   */
  async deleteMeetLink(appointmentId, meetLink) {
    if (!this.initialized) {
      return true; // Nothing to delete for fallback links
    }

    try {
      // Extract event ID from meet link or use appointmentId
      // This requires storing the eventId when creating the event
      // For now, we'll just log
      console.log('🗑️  Deleting meet link for appointment:', appointmentId);
      return true;
    } catch (error) {
      console.error('❌ Error deleting meet link:', error.message);
      return false;
    }
  }
}

// Export singleton instance
const googleMeetService = new GoogleMeetService();

module.exports = { googleMeetService };
