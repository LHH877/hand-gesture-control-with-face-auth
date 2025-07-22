#------------------
# Main.py
#------------------

import os
import cv2
import pickle
import numpy as np
import face_recognition
import firebase_admin
from firebase_admin import credentials, db, storage
import tkinter as tk
from threading import Thread
import time
from datetime import datetime  # Added for timestamp formatting

# Hand Tracking Module (incorporated directly)
class HandDetector:
    """
    A class to perform hand detection and landmark identification using MediaPipe.
    """
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initializes the hand detector with given parameters.
        """
        import mediapipe as mp  # Import MediaPipe for hand tracking
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands  # Initialize the MediaPipe hands module
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.max_hands, 
            min_detection_confidence=self.detection_confidence, 
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils  # Utility for drawing hand landmarks

    def find_hands(self, image, draw=True):
        """
        Detects hands in an image.
        Parameters:
            image: The input image.
            draw: Whether to draw hand landmarks.
        Returns:
            image: The output image with or without drawings.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        self.results = self.hands.process(image_rgb)  # Detect hands

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)  # Draw hand landmarks
        return image

    def find_position(self, image, hand_no=0, draw=True):
        """
        Finds the positions of hand landmarks.
        Parameters:
            image: The input image.
            hand_no: The index of the hand to process.
            draw: Whether to draw circles on the landmarks.
        Returns:
            landmark_list: A list of landmark positions.
        """
        landmark_list = []
        if self.results.multi_hand_landmarks:
            try:
                my_hand = self.results.multi_hand_landmarks[hand_no]  # Get the specified hand
                for idx, lm in enumerate(my_hand.landmark):
                    img_height, img_width, img_channel = image.shape
                    coord_x, coord_y = int(lm.x * img_width), int(lm.y * img_height)  # Calculate pixel coordinates
                    landmark_list.append([idx, coord_x, coord_y])
                    if draw:
                        cv2.circle(image, (coord_x, coord_y), 5, (255, 0, 255), cv2.FILLED)  # Draw a circle on the landmark
            except IndexError:
                pass  # Handle the case when hand_no is out of range
        return landmark_list

# Firebase and Face Recognition Code
def initialize_firebase():
    """Initialize the Firebase application."""
    cred = credentials.Certificate("serviceAccountKey.json")  # Load the Firebase credentials
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://finalsp-d001a-default-rtdb.firebaseio.com/",  # Connect to Firebase Realtime Database
        'storageBucket': "finalsp-d001a.appspot.com"  # Connect to Firebase Storage
    })
    return storage.bucket()  # Return the Firebase Storage bucket

def load_encodings():
    """Load face encodings from the file."""
    print("Loading Encodings File...")
    with open("EncodeFile.p", "rb") as file:
        encodings_with_ids = pickle.load(file)  # Load face encodings from the file
    known_encodings, user_ids = encodings_with_ids  # Separate encodings and user IDs
    print("Encodings File Loaded")
    return known_encodings, user_ids

def setup_camera():
    """Set up the video capture device."""
    camera = cv2.VideoCapture(0)  # Use the default camera (index 0)
    camera.set(3, 640)  # Set camera width to 640
    camera.set(4, 480)  # Set camera height to 480
    return camera

def load_overlay_images(folder_path):
    """
    Loads overlay images from the specified folder.
    Parameters:
        folder_path: Path to the folder containing images.
    Returns:
        overlay_images: List of loaded images.
    """
    image_list = os.listdir(folder_path)  # List all images in the folder
    overlay_images = []
    for img_path in image_list:
        image = cv2.imread(f'{folder_path}/{img_path}')  # Read each image
        overlay_images.append(image)  # Add the image to the list
    return overlay_images

# Function to log user activity
def log_user_activity(user_id, activity):
    """
    Logs user activity to Firebase with a readable timestamp.
    Parameters:
        user_id: The ID of the user.
        activity: Description of the activity.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current timestamp
    ref = db.reference(f'UserActivity/{user_id}')  # Reference the user’s activity in Firebase
    ref.push({
        'activity': activity,  # Log the activity
        'timestamp': timestamp  # Log the timestamp
    })

def recognize_faces():
    """Main function to recognize faces and then switch to finger counting."""
    global running, current_user_id  # Access global variables
    bucket = initialize_firebase()  # Initialize Firebase
    camera = setup_camera()  # Set up the camera
    known_encodings, user_ids = load_encodings()  # Load face encodings

    current_user_id = None
    user_info = None
    user_image = None
    frame_count = 0
    frame_skip = 5  # Process every 5th frame to reduce load

    font_type = cv2.FONT_HERSHEY_COMPLEX  # Font settings for the display
    font_scale = 1.0
    font_color = (0, 255, 0)  # Green color text

    start_time = None
    switch_to_finger_counter = False

    while running:
        if switch_to_finger_counter:
            # Switch to finger counting after 5 seconds of face recognition
            finger_counter()
            return

        success, frame = camera.read()  # Read the camera feed
        if not success:
            print("Failed to read from camera.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip processing frames based on frame_skip

        status = 'Active'
        authorized_user_detected = False

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resize frame for faster processing
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert the image to RGB

        face_locations = face_recognition.face_locations(rgb_small_frame)  # Detect face locations
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Generate face encodings

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)  # Compare faces
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)  # Calculate distances
            best_match_index = np.argmin(face_distances)  # Get the closest match

            if matches[best_match_index]:
                user_id = user_ids[best_match_index]  # Get the user ID
                status = 'Detected'
                authorized_user_detected = True

                if current_user_id != user_id:
                    current_user_id = user_id  # Update the current user ID
                    user_info = db.reference(f'Authorized Users/{user_id}').get()  # Get user info from Firebase
                    # Get the user's image from Firebase storage
                    blob = bucket.get_blob(f'Images/{user_id}.png')
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    user_image = cv2.imdecode(array, cv2.IMREAD_COLOR)  # Decode the image
                    user_image = cv2.resize(user_image, (150, 150))  # Resize the image
                    
                    log_user_activity(current_user_id, "User Logged In")  # Log the user login activity
                    
                    print("Known Face Detected")  # Print a success message
                    print(user_ids[best_match_index])

                # Start a timer when an authorized user is detected
                if start_time is None:
                    start_time = time.time()

        if not authorized_user_detected:
            if current_user_id is not None:
                log_user_activity(current_user_id, "User Logged Out")  # Log the user logout activity
            current_user_id = None  # Reset user info
            user_info = None
            user_image = None
            start_time = None

        if status == 'Detected' and current_user_id and user_info and user_image is not None:
            # Set position to display the user image at the bottom left
            x_offset = 10
            y_offset = frame.shape[0] - user_image.shape[0] - 10  # Adjust the y_offset for bottom-left placement

            frame[y_offset:y_offset+user_image.shape[0], x_offset:x_offset+user_image.shape[1]] = user_image  # Overlay user image

            # Display the user's name and department below the image
            name_text = user_info['name']
            dept_text = user_info['Department']

            name_y_offset = y_offset + user_image.shape[0] + 30  # Calculate y_offset for name
            dept_y_offset = name_y_offset + 30  # Calculate y_offset for department

            # Adjust the text if it goes out of bounds
            if dept_y_offset > frame.shape[0] - 10:
                name_y_offset = frame.shape[0] - 60
                dept_y_offset = frame.shape[0] - 30

            # Center the text in the frame
            text_width, _ = cv2.getTextSize(name_text, font_type, 0.8, 2)[0]
            text_x_offset = (frame.shape[1] - text_width) // 2

            # Display the name and department
            cv2.putText(frame, name_text, (text_x_offset, name_y_offset), font_type, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, dept_text, (text_x_offset, dept_y_offset), font_type, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)  # Show the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
            running = False

        # Switch to finger counting after 5 seconds
        if start_time is not None and time.time() - start_time > 5:
            switch_to_finger_counter = True

    camera.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

def finger_counter():
    """Function to run the finger counting app."""
    global current_user_id
    width_camera, height_camera = 640, 480  # Set camera resolution
    cap = cv2.VideoCapture(0)  # Open the camera
    cap.set(3, width_camera)
    cap.set(4, height_camera)

    folder_path = "FingerImages"  # Path for overlay images
    overlay_images = load_overlay_images(folder_path)  # Load overlay images for fingers

    previous_time = 0
    frame_counter = 0
    frame_skip = 2  # Process every 2 frames to reduce load

    detector = HandDetector(detection_confidence=0.75)  # Initialize hand detector
    tip_ids = [4, 8, 12, 16, 20]  # List of landmark IDs for fingertips

    previous_finger_count = None  # To track gesture changes

    while True:
        success, image = cap.read()  # Read from camera
        if not success:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue  # Skip frame processing

        image = detector.find_hands(image)  # Detect hands
        landmark_list = detector.find_position(image, draw=False)  # Find hand positions

        if len(landmark_list) != 0:
            fingers = []

            # Thumb detection logic
            if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for idx in range(1, 5):
                if landmark_list[tip_ids[idx]][2] < landmark_list[tip_ids[idx] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)  # Count the number of extended fingers

            # Log finger gesture activity if it changes
            if previous_finger_count is None:
                previous_finger_count = total_fingers
                log_user_activity(current_user_id, f"Gesture detected: {total_fingers} fingers")
            elif total_fingers != previous_finger_count:
                log_user_activity(current_user_id, f"Gesture changed from {previous_finger_count} to {total_fingers} fingers")
                previous_finger_count = total_fingers

            # Overlay corresponding image for the number of fingers
            if 0 < total_fingers <= len(overlay_images):
                h, w, c = overlay_images[total_fingers - 1].shape
                image[0:h, 0:w] = overlay_images[total_fingers - 1]

        cv2.imshow("Finger Counting", image)  # Display the image
        if cv2.waitKey(1) == ord('q'):  # Exit the loop if 'q' is pressed
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

def start_recognition():
    """Start the face recognition in a separate thread."""
    global running
    running = True
    Thread(target=recognize_faces).start()  # Run the face recognition in a separate thread

# Function to display the consent form
def display_consent_form(app):
    """
    Displays the consent form for users to agree or disagree before using the system.
    Parameters:
        app: The main Tkinter application window.
    """
    app.withdraw()  # Hide the main window while the consent form is active

    consent_window = tk.Toplevel(app)  # Create a new window for the consent form
    consent_window.title("Informed Consent Form")
    
    # Text for the consent form
    consent_text = (
        "Informed Consent Form\n\n"
        "Introduction\n"
        "Welcome to the Hand Gesture Recognition Control System secured by Facial Recognition.\n"
        "Before proceeding to use the system, we kindly ask for your consent to collect and "
        "process your data for the purpose of system authentication and interaction.\n\n"
        "Data Collected\n"
        "- Facial images for recognition and authentication.\n"
        "- Hand gesture data for control and interaction.\n"
        "- Activity logs, including timestamps of actions within the system.\n\n"
        "Purpose of Data Collection\n"
        "Your facial data and hand gesture data are collected solely for the purpose of enabling "
        "secure access and interaction with the system. Your data will not be shared with third "
        "parties for marketing or any purposes outside the system’s operation.\n\n"
        "Storage and Security\n"
        "Your data is securely stored using Firebase and encrypted to protect against unauthorized access.\n"
        "It will only be accessible by system administrators and developers to ensure proper functionality.\n\n"
        "Your Rights\n"
        "You have the right to access your data, request its deletion, or withdraw your consent at any time.\n"
        "If you wish to exercise these rights, please contact us.\n\n"
        "By clicking 'I Agree,' you consent to the collection, storage, and use of your data as outlined above."
    )
    
    
    # Create a label widget to display the consent text
    consent_label = tk.Label(consent_window, text=consent_text, wraplength=500, justify="left")
    consent_label.pack(padx=10, pady=10)

    # Function to proceed when the user agrees
    def agree_action():
        consent_window.destroy()  # Close the consent window
        start_recognition()  # Start the face recognition system
    
    # Function to exit when the user disagrees
    def disagree_action():
        app.quit()  # Exit the application

    # Create buttons for Agree and Disagree
    agree_button = tk.Button(consent_window, text="I Agree", command=agree_action, width=25, height=2)
    disagree_button = tk.Button(consent_window, text="I Disagree", command=disagree_action, width=25, height=2)
    
    # Pack buttons into the consent window
    agree_button.pack(pady=10)
    disagree_button.pack(pady=10)

# GUI Interface using Tkinter
if __name__ == "__main__":
    running = False
    current_user_id = None  # Initialize as None
    app = tk.Tk()  # Create the main Tkinter window
    app.title("Face Recognition and Finger Counting App")

    # Display the consent form first before starting the recognition system
    display_consent_form(app)

    app.mainloop()  # Start the Tkinter event loop
