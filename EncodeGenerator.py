#-----------------------------
# EncodeGenerator.py
#-----------------------------


import os
import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase and return the storage bucket
def initialize_firebase():
    """
    Initialize the Firebase application with credentials and connect to the database and storage.
    """
    cred = credentials.Certificate("serviceAccountKey.json")  # Load Firebase credentials
    firebase_admin.initialize_app(cred, {
        'databaseURL': "(Input your own firebase database URL)",  # Connect to the Realtime Database
        'storageBucket': "(Input your own firebase storage URL)"  # Connect to Firebase storage
    })
    return storage.bucket()  # Return the storage bucket reference

# Load images from the specified folder and extract user IDs
def load_images(folder_path):
    """
    Load images from the specified folder, and extract the corresponding user IDs from filenames.
    """
    image_paths = os.listdir(folder_path)  # Get a list of all image file names in the folder
    images = []  # List to store loaded images
    user_ids = []  # List to store user IDs (extracted from filenames)

    # Load each image and extract the user ID from the filename (excluding the file extension)
    for path in image_paths:
        image = cv2.imread(os.path.join(folder_path, path))  # Load the image
        images.append(image)  # Add the image to the list
        user_ids.append(os.path.splitext(path)[0])  # Extract user ID from the filename

    return images, user_ids, image_paths  # Return the images, user IDs, and image file paths

# Upload images to Firebase Storage
def upload_images_to_firebase(bucket, folder_path, image_paths):
    """
    Upload the images to Firebase Storage, storing them with the same filenames.
    """
    for path in image_paths:
        file_name = f'{folder_path}/{path}'  # Full file path for the image
        blob = bucket.blob(file_name)  # Create a new storage blob (file) in the Firebase bucket
        blob.upload_from_filename(file_name)  # Upload the image to Firebase
        print(f"Uploaded {file_name} to Firebase Storage.")  # Print success message for each upload

# Find face encodings for the loaded images
def find_encodings(images_list):
    """
    Generate face encodings for the given list of images using the face_recognition library.
    """
    encoding_list = []  # List to store face encodings
    for image in images_list:
        try:
            # Convert the image from BGR (OpenCV format) to RGB (required by face_recognition)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Generate the face encoding for the image (only if face is detected)
            encoding = face_recognition.face_encodings(image_rgb)[0]
            encoding_list.append(encoding)  # Add the encoding to the list
        except IndexError as e:
            # Handle the case where no face is detected in the image
            print(f"Face encoding failed for an image: {e}")
            continue  # Skip this image and continue with the next one
    return encoding_list  # Return the list of face encodings

# Save the face encodings and corresponding user IDs to a pickle file
def save_encodings(encoding_list, user_ids):
    """
    Save the face encodings along with user IDs into a file using pickle for later use.
    """
    encodings_with_ids = [encoding_list, user_ids]  # Combine encodings and user IDs into a single list
    with open("EncodeFile.p", "wb") as file:  # Open a file in binary write mode
        pickle.dump(encodings_with_ids, file)  # Save the data into the file using pickle
    print("Encodings saved to EncodeFile.p")  # Print success message when encoding is saved

# Main function that orchestrates the process
def main():
    """
    Main function to initialize Firebase, load images, upload them to Firebase, generate encodings,
    and save the encodings along with user IDs.
    """
    bucket = initialize_firebase()  # Initialize Firebase and get the storage bucket
    folder_path = 'Images'  # Define the folder path where images are stored
    images, user_ids, image_paths = load_images(folder_path)  # Load images and user IDs
    upload_images_to_firebase(bucket, folder_path, image_paths)  # Upload the images to Firebase storage
    print("Encoding Started...")  # Notify that encoding is starting
    encoding_list = find_encodings(images)  # Generate face encodings from the loaded images
    print("Encoding Complete")  # Notify when encoding is completed
    save_encodings(encoding_list, user_ids)  # Save the encodings along with user IDs

# Entry point of the script
if __name__ == "__main__":
    main()  # Run the main function
