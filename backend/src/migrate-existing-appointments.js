/**
 * Firebase Migration Script
 * Updates all existing appointments with working Jitsi Meet links
 * 
 * HOW TO RUN:
 * 1. Make sure you have Firebase initialized in your project
 * 2. Run: node backend/src/migrate-existing-appointments.js
 */

const admin = require('firebase-admin');
const path = require('path');

// Initialize Firebase Admin SDK
// You'll need your Firebase service account key
const serviceAccount = require('../firebase-service-account.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

async function migrateAppointments() {
  console.log('\n' + '='.repeat(60));
  console.log('🚀 Starting Appointment Migration to Jitsi Meet');
  console.log('='.repeat(60) + '\n');

  try {
    // Get all appointments
    const appointmentsRef = db.collection('appointments');
    const snapshot = await appointmentsRef.get();

    if (snapshot.empty) {
      console.log('❌ No appointments found in Firestore');
      return;
    }

    console.log(`📋 Found ${snapshot.size} appointments to migrate\n`);

    let updated = 0;
    let skipped = 0;
    let errors = 0;

    for (const doc of snapshot.docs) {
      const data = doc.data();
      const appointmentId = doc.id;
      const oldMeetLink = data.meetLink || 'N/A';

      try {
        // Check if it's a fake Google Meet link
        if (oldMeetLink.includes('meet.google.com')) {
          // Generate new Jitsi Meet link
          const timestamp = Date.now() + updated; // Ensure uniqueness
          const newMeetLink = `https://meet.jit.si/ongogenesis-${appointmentId}-${timestamp}`;

          // Update the appointment
          await doc.ref.update({
            meetLink: newMeetLink,
            updatedAt: admin.firestore.FieldValue.serverTimestamp(),
            migratedToJitsi: true,
            oldMeetLink: oldMeetLink // Keep old link for reference
          });

          console.log(`✅ [${updated + 1}/${snapshot.size}] ${appointmentId}`);
          console.log(`   Old: ${oldMeetLink}`);
          console.log(`   New: ${newMeetLink}\n`);

          updated++;
        } else if (oldMeetLink.includes('meet.jit.si')) {
          console.log(`⏭️  [${updated + skipped + 1}/${snapshot.size}] ${appointmentId} - Already using Jitsi\n`);
          skipped++;
        } else {
          console.log(`⚠️  [${updated + skipped + 1}/${snapshot.size}] ${appointmentId} - Unknown format: ${oldMeetLink}\n`);
          skipped++;
        }
      } catch (error) {
        console.error(`❌ Error updating ${appointmentId}:`, error.message);
        errors++;
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log('📊 Migration Complete!');
    console.log('='.repeat(60));
    console.log(`✅ Updated: ${updated} appointments`);
    console.log(`⏭️  Skipped: ${skipped} appointments`);
    console.log(`❌ Errors: ${errors} appointments`);
    console.log('='.repeat(60) + '\n');

    process.exit(0);

  } catch (error) {
    console.error('\n❌ Migration failed:', error);
    process.exit(1);
  }
}

// Run migration
migrateAppointments();
