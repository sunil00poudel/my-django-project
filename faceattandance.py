import tkinter as tk
from tkinter import simpledialog
import cv2
from PIL import Image, ImageTk
import os
import face_recognition
from datetime import datetime
import csv
import pyttsx3

# --- INITIALIZATION ---
# Initialize the Voice Engine
engine = pyttsx3.init()

# --- GUI SETUP ---
root = tk.Tk()
root.title('Attendance Sheet System')
root.minsize(800, 700)
root.resizable(False, False)

# Main Video Display
video_label = tk.Label(root)  
video_label.pack()

# Control Panel (The Button Container)
control_frame = tk.Frame(root)
control_frame.pack(side='bottom', fill='x', pady=20)

# Video Capture Object
cap = cv2.VideoCapture(0)

# --- FUNCTIONS ---

def register_new_user():
    """Captures one frame and saves it in a folder named after the User ID."""
    user_id = simpledialog.askstring("Input", "Enter User Name or ID:")
    
    if user_id:
        ret, frame = cap.read()
        if ret:
            if not os.path.exists('training_data'):
                os.makedirs('training_data')

            user_path = os.path.join('training_data', user_id)
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            
            # Save raw BGR frame for high quality
            img_name = os.path.join(user_path, f"{user_id}.jpg")
            cv2.imwrite(img_name, frame)
            
            print(f"User {user_id} registered successfully.")
            engine.say(f"Registration complete for {user_id}")
            engine.runAndWait()
        else:
            print("Error: Could not access camera.")

def process_attendance(status):
    """Compares live video frame to all images in training_data."""
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture the image')
        return

    # 1. Prepare the live face
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        print('No face detected! Please look at the camera.')
        engine.say("Face not detected")
        engine.runAndWait()
        return
    
    live_encoding = face_encodings[0] 
    match_found = False 

    if not os.path.exists('training_data'):
        print("No database found. Please register a user first.")
        return

    # 2. Compare against every user in the database
    for user_id in os.listdir('training_data'):
        img_path = os.path.join('training_data', user_id, f"{user_id}.jpg")
        
        if os.path.exists(img_path):
            known_image = face_recognition.load_image_file(img_path)
            known_list = face_recognition.face_encodings(known_image)
            
            if len(known_list) > 0:
                known_encoding = known_list[0]
                # Compare live vs saved
                results = face_recognition.compare_faces([known_encoding], live_encoding)

                if results[0]:
                    match_found = True
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Logic for Speech and Logging
                    message = f"Thank you {user_id}, {status} successful"
                    print(message)
                    
                    # Log to CSV
                    with open('attendance.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([user_id, status, now])

                    # Audio Feedback
                    engine.say(message)
                    engine.runAndWait()
                    break 

    if not match_found:
        print('User is not registered.')
        engine.say("User not recognized")
        engine.runAndWait()

def update_frame():
    """The continuous loop that refreshes the GUI video feed."""
    ret, frame = cap.read()
    if ret:
        # Flip and resize for the GUI (Mirror effect)
        display_frame = cv2.flip(frame, 1)
        display_frame = cv2.resize(display_frame, (800, 600))
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(display_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.config(image=img_tk)
        video_label.img_tk = img_tk # type: ignore

    # Repeat every 10 milliseconds
    video_label.after(10, update_frame)

# --- UI BUTTON WIRING ---
btn_check_in = tk.Button(control_frame, text='Check In', width=15, 
                         command=lambda: process_attendance('check in'))
btn_check_in.pack(side='left', padx=10, expand=True)

btn_check_out = tk.Button(control_frame, text='Check Out', width=15, 
                          command=lambda: process_attendance('check out'))
btn_check_out.pack(side='left', padx=10, expand=True)

btn_new_user = tk.Button(control_frame, text='New User', width=15, 
                         command=register_new_user)
btn_new_user.pack(side='right', padx=10, expand=True)

# --- LAUNCH ---
update_frame()
root.mainloop()

# Clean up
cap.release()