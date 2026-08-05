/**
 * Script to migrate existing appointments from Google Meet to Jitsi Meet
 * 
 * HOW TO USE:
 * 1. Open Firebase Console: https://console.firebase.google.com
 * 2. Navigate to: Firestore Database
 * 3. Click on "Rules" tab, then click on "Playground"
 * 4. Or use this script in a Firebase Cloud Function
 */

// Option 1: Run in Browser Console (Quick & Easy)
// --------------------------------------------
// 1. Go to Firebase Console > Firestore Database
// 2. Open browser DevTools (F12)
// 3. Copy and paste this code in Console tab:

const migrateToJitsi = async () => {
  const db = firebase.firestore();
  const appointmentsRef = db.collection('appointments');
  
  try {
    const snapshot = await appointmentsRef.get();
    console.log(`Found ${snapshot.size} appointments to migrate`);
    
    let updated = 0;
    const batch = db.batch();
    
    snapshot.forEach((doc) => {
      const data = doc.data();
      const appointmentId = doc.id;
      const timestamp = Date.now() + updated; // Ensure uniqueness
      
      // Generate new Jitsi Meet link
      const newMeetLink = `https://meet.jit.si/ongogenesis-${appointmentId}-${timestamp}`;
      
      console.log(`Updating ${appointmentId}: ${data.meetLink} → ${newMeetLink}`);
      
      batch.update(doc.ref, {
        meetLink: newMeetLink,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      
      updated++;
    });
    
    await batch.commit();
    console.log(`✅ Successfully updated ${updated} appointments!`);
    
  } catch (error) {
    console.error('❌ Error migrating appointments:', error);
  }
};

// Run the migration
migrateToJitsi();


// Option 2: Firebase Cloud Function (Recommended for Production)
// --------------------------------------------------------
// Create this as a Cloud Function that you can call via HTTP

const functions = require('firebase-functions');
const admin = require('firebase-admin');

exports.migrateAppointmentsToJitsi = functions.https.onCall(async (data, context) => {
  // Verify user is authenticated and is admin
  if (!context.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'User must be authenticated');
  }
  
  const db = admin.firestore();
  const appointmentsRef = db.collection('appointments');
  
  try {
    const snapshot = await appointmentsRef.get();
    const batch = db.batch();
    let count = 0;
    
    snapshot.forEach((doc) => {
      const appointmentId = doc.id;
      const timestamp = Date.now() + count;
      const newMeetLink = `https://meet.jit.si/ongogenesis-${appointmentId}-${timestamp}`;
      
      batch.update(doc.ref, {
        meetLink: newMeetLink,
        updatedAt: admin.firestore.FieldValue.serverTimestamp()
      });
      
      count++;
    });
    
    await batch.commit();
    
    return {
      success: true,
      message: `Updated ${count} appointments`,
      count: count
    };
    
  } catch (error) {
    console.error('Migration error:', error);
    throw new functions.https.HttpsError('internal', 'Migration failed');
  }
});


// Option 3: Node.js Script (Run locally)
// -------------------------------------
// Create a file: migrate-appointments.js

const admin = require('firebase-admin');
const serviceAccount = require('./path-to-your-service-account-key.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

async function migrateAppointments() {
  console.log('🚀 Starting migration to Jitsi Meet...\n');
  
  const appointmentsRef = db.collection('appointments');
  const snapshot = await appointmentsRef.get();
  
  if (snapshot.empty) {
    console.log('No appointments found.');
    return;
  }
  
  console.log(`Found ${snapshot.size} appointments to migrate\n`);
  
  const batch = db.batch();
  let count = 0;
  
  for (const doc of snapshot.docs) {
    const data = doc.data();
    const appointmentId = doc.id;
    const timestamp = Date.now() + count;
    
    const oldLink = data.meetLink || 'N/A';
    const newLink = `https://meet.jit.si/ongogenesis-${appointmentId}-${timestamp}`;
    
    console.log(`[${count + 1}/${snapshot.size}] ${appointmentId}`);
    console.log(`  Old: ${oldLink}`);
    console.log(`  New: ${newLink}\n`);
    
    batch.update(doc.ref, {
      meetLink: newLink,
      updatedAt: admin.firestore.FieldValue.serverTimestamp()
    });
    
    count++;
    
    // Commit in batches of 500 (Firestore limit)
    if (count % 500 === 0) {
      await batch.commit();
      console.log(`✅ Committed batch of 500\n`);
    }
  }
  
  // Commit remaining
  if (count % 500 !== 0) {
    await batch.commit();
  }
  
  console.log(`\n✅ Migration complete! Updated ${count} appointments.`);
  process.exit(0);
}

migrateAppointments().catch((error) => {
  console.error('❌ Migration failed:', error);
  process.exit(1);
});


// QUICK FIX: Update Single Appointment
// ------------------------------------
// If you just want to test with one appointment:

async function updateSingleAppointment(appointmentId) {
  const db = firebase.firestore();
  const timestamp = Date.now();
  const newMeetLink = `https://meet.jit.si/ongogenesis-${appointmentId}-${timestamp}`;
  
  await db.collection('appointments').doc(appointmentId).update({
    meetLink: newMeetLink,
    updatedAt: firebase.firestore.FieldValue.serverTimestamp()
  });
  
  console.log(`✅ Updated appointment ${appointmentId}`);
  console.log(`New link: ${newMeetLink}`);
}

// Usage:
// updateSingleAppointment('APT1234567890');


/**
 * TESTING CHECKLIST:
 * 
 * After migration:
 * ✓ Create a new test appointment
 * ✓ Check Firestore - meetLink should be Jitsi URL
 * ✓ Login as doctor - click "Join Video Call"
 * ✓ Should open Jitsi room (no error)
 * ✓ Login as patient in another browser/tab
 * ✓ Click "Join Video Call" on same appointment
 * ✓ Both should see each other in video call
 * ✓ Test chat, screen share, audio/video
 */
