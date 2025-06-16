from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from scipy.spatial import distance as dist
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
import mediapipe as mp

app = Flask(__name__)

EYE_AR_THRESHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 3

LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'Attendance.csv'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

latest_marked_name = None
tolerance = 0.5
marked_attendance = set()
known_faces_encodings = []
known_names_list = []

cached_boxes = []
cached_names = []

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load known faces
def load_known_faces():
    global known_faces_encodings, known_names_list
    known_faces_encodings = []
    known_names_list = []
    for name in os.listdir(KNOWN_FACES_DIR):
        if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name)):
            for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
                image_path = os.path.join(KNOWN_FACES_DIR, name, filename)
                if os.path.isfile(image_path):
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_faces_encodings.append(encodings[0])
                            known_names_list.append(name)
                    except Exception as e:
                        print(f"[ERROR] Failed to process {image_path}: {e}")

# Mark attendance
def mark(name):
    with open(ATTENDANCE_FILE, 'a') as f:
        now = datetime.now()
        f.write(f'{name},Present,{now.strftime("%Y-%m-%d %H:%M:%S")}\n')

# Mark on blink
def mark_attendance_with_blink(name, blink_detected):
    global marked_attendance, latest_marked_name
    if blink_detected and name not in marked_attendance:
        mark(name)
        marked_attendance.add(name)
        latest_marked_name = name
        print(f"[INFO] Attendance marked for {name}!")

def process_frame(frame, skip_detection=False):
    global cached_boxes, cached_names
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ih, iw, _ = frame.shape

    if skip_detection:
        # Draw cached boxes
        for (top, right, bottom, left), name in zip(cached_boxes, cached_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame

    face_locations = []
    face_encodings = []
    try:
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_coords = [lm.x * iw for lm in face_landmarks.landmark]
                y_coords = [lm.y * ih for lm in face_landmarks.landmark]
                left, right = int(min(x_coords)), int(max(x_coords))
                top, bottom = int(min(y_coords)), int(max(y_coords))
                # Expand bounding box
                box_width, box_height = right-left, bottom-top
                expand_w, expand_h = int(box_width*0.1), int(box_height*0.1)
                left, right = max(left-expand_w,0), min(right+expand_w,iw)
                top, bottom = max(top-expand_h,0), min(bottom+expand_h,ih)
                face_locations.append((top,right,bottom,left))

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception as e:
        print(f"[ERROR] Face mesh processing: {e}")

    boxes, names = [], []
    for (top,right,bottom,left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces_encodings, encoding, tolerance=tolerance)
        name = "Unknown"
        if True in matches:
            name = known_names_list[matches.index(True)]
            blink_detected = False
            try:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    left_eye = [(int(lm.x*iw), int(lm.y*ih)) for lm in [landmarks[i] for i in LEFT_EYE_LANDMARKS]]
                    right_eye = [(int(lm.x*iw), int(lm.y*ih)) for lm in [landmarks[i] for i in RIGHT_EYE_LANDMARKS]]
                    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye))/2
                    if ear < EYE_AR_THRESHOLD:
                        blink_detected = True
                    break
            except Exception as e:
                print(f"[ERROR] Blink detection: {e}")
            mark_attendance_with_blink(name, blink_detected)
        boxes.append((top,right,bottom,left))
        names.append(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left+6,bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cached_boxes, cached_names = boxes, names
    return frame




def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + open('static/error.jpg', 'rb').read() + b'\r\n'
        return

    frame_count = 0
    detection_interval = 3 
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Failed to capture frame")
                break

            # Decide whether to skip detection
            skip_detection = frame_count % detection_interval != 0
            processed_frame = process_frame(frame, skip_detection=skip_detection)
            frame_count += 1

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
    finally:
        cap.release()


def add_new_face(name, image_data):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    image_path = os.path.join(person_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img.save(image_path)
        image_np = np.array(img)
        encodings = face_recognition.face_encodings(image_np)
        if encodings:
            return encodings[0], name
        else:
            os.remove(image_path)
            return None, None
    except Exception as e:
        print(f"[ERROR] Failed to save or encode new face: {e}")
        return None, None

def visualize_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE, names=["Name", "Status", "Timestamp"])
        
        # Bar chart: Attendance Count by Name
        attendance_summary = df.groupby("Name").size().reset_index(name="Count")
        fig_bar = px.bar(
            attendance_summary,
            x="Name", y="Count", color="Count",
            title="Attendance Count by Name",
            labels={"Name": "Name", "Count": "Attendance Count"},
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(
            autosize=True,
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(tickangle=-45),
        )
        
        # Line chart: Daily Attendance Trend
        df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
        daily_summary = df.groupby("Date").size().reset_index(name="Count")
        fig_line = px.line(
            daily_summary,
            x="Date", y="Count",
            title="Daily Attendance Trend",
            labels={"Date": "Date", "Count": "Attendance Count"},
            markers=True
        )
        fig_line.update_layout(
            autosize=True,
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(tickangle=-45),
        )
        
        return (
            fig_bar.to_html(full_html=False, include_plotlyjs='cdn', config={"responsive": True}),
            fig_line.to_html(full_html=False, include_plotlyjs='cdn', config={"responsive": True}),
            df.to_html(index=False, classes='dataframe'),
            sorted(df['Name'].unique())
        )
    return None, None, "<p>Attendance file not found!</p>", []


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_face_page')
def add_face_page():
    return render_template('add_face.html')

@app.route('/video_feed')
@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + open('static/error.jpg', 'rb').read() + b'\r\n'
            return
        
        frame_count = 0
        detection_interval = 3
        
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Failed to capture frame")
                continue  # Instead of breaking, retry

            skip_detection = frame_count % detection_interval != 0
            try:
                processed_frame = process_frame(frame, skip_detection=skip_detection)
            except Exception as e:
                print(f"[ERROR] Processing frame: {e}")
                processed_frame = frame  # fallback to original frame

            frame_count += 1
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_attendance')
def latest_attendance():
    global latest_marked_name
    if latest_marked_name:
        name = latest_marked_name
        latest_marked_name = None
        return jsonify({'marked': True, 'name': name})
    return jsonify({'marked': False})

@app.route('/add_face', methods=['POST'])
def add_face():
    name = request.form['name']
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    if file:
        image_data = file.read()
        encoding, new_name = add_new_face(name, image_data)
        if encoding is not None and new_name:
            load_known_faces()
            return jsonify({'success': f'Added {new_name} successfully!'})
        else:
            return jsonify({'error': 'Failed to add face. Try again with a clear image.'}), 400
    return jsonify({'error': 'An error occurred'}), 500

@app.route('/add_captured_face', methods=['POST'])
def add_captured_face():
    data = request.get_json()
    name = data.get('name')
    image_data = data.get('image')
    if not name or not image_data:
        return jsonify({'error': 'Missing name or image data'}), 400
    try:
        header, encoded = image_data.split(',', 1)
        binary_data = base64.b64decode(encoded)
        encoding, new_name = add_new_face(name, binary_data)
        if encoding is not None and new_name:
            load_known_faces()
            return jsonify({'success': f'{new_name} added successfully!'})
        else:
            return jsonify({'error': 'Face not detected. Try again with a clear image.'}), 400
    except Exception as e:
        print(f"[ERROR] Failed to process captured face: {e}")
        return jsonify({'error': 'Error processing image'}), 500

@app.route('/visualize')
def visualize():
    bar_chart, line_chart, records, students = visualize_attendance()
    return render_template('visualization.html', bar_chart=bar_chart, line_chart=line_chart, records=records, students=students)

@app.route('/visualize/student', methods=['POST'])
def visualize_student():
    name = request.form['student_name']
    if not os.path.exists(ATTENDANCE_FILE):
        return render_template('visualization.html', bar_chart=None, line_chart=None, records="<p>Attendance file not found!</p>", students=[])

    df = pd.read_csv(ATTENDANCE_FILE, names=["Name", "Status", "Timestamp"])
    df = df[df['Name'] == name]
    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
    grouped = df.groupby("Date").size().reset_index(name="Count")
    fig_line = px.line(grouped, x="Date", y="Count", title=f"Attendance Trend for {name}", labels={"Date": "Date", "Count": "Attendance Count"}, markers=True)
    students = sorted(pd.read_csv(ATTENDANCE_FILE, names=["Name", "Status", "Timestamp"])['Name'].unique())
    return render_template('visualization.html', bar_chart=None, line_chart=fig_line.to_html(full_html=False), records=df.to_html(index=False, classes='dataframe'), students=students)


if __name__ == '__main__':
    load_known_faces()
    app.run(debug=False)
