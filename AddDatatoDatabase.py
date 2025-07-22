#-----------------------------
# AddDatatoDatabase.py
#-----------------------------

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase with the credentials and connect to the Realtime Database
cred = credentials.Certificate("serviceAccountKey.json")  # Load the Firebase credentials from the service account key file
firebase_admin.initialize_app(cred, {
    'databaseURL': "(Input your own firebase database URL)"  # Connect to the Firebase Realtime Database
})

# Reference the 'Authorized Users' node in the database
ref = db.reference('Authorized Users')  # Create a reference to the 'Authorized Users' section in the Firebase database

# Dictionary containing data to be added to the database
data = {
    "Bill Gates":  # Key in the database is the user ID (in this case, a person's name)
        {
            "name": "User 1",  # User's name
            "Department": "IT"  # User's department
        },
    "Elon Musk":
        {
            "name": "User 2",
            "Department": "Sales"
        },
    "Lim Heng Hoe":
        {
            "name": "User 3",
            "Department": "Marketing"
        }

    ,
    "Jeff Bezos":
        {
            "name": "User 4",
            "Department": "HR"
        }
}

# Add each user and their information to the database
for key, value in data.items():
    ref.child(key).set(value)  # For each key (user ID), set the corresponding value (user details) in the Firebase database
