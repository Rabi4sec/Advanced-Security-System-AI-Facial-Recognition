import os
import time
import threading
import queue
import numpy as np
import cv2
import face_recognition
import subprocess
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from flask_socketio import SocketIO
from functools import wraps
from models import User, Face, Alert
from extensions import db
import datetime
from base64 import b64encode

# Initialize Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Path to FFmpeg executable
ffmpeg_path =  'ffmpeg'

# Known face data
known_face_encodings = []
known_face_names = []

# Face Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Alert management
last_alert_times = {}
ALERT_COOLDOWN = 5  # seconds

# Alert queue
alert_queue = queue.Queue()

# ========== HELPERS ==========

def init_db():
    with app.app_context():
        db.create_all()

        default_username = os.getenv('ADMIN_USERNAME')
        default_password = os.getenv('ADMIN_PASSWORD')

        if not default_username or not default_password:
            raise ValueError("ADMIN_USERNAME or ADMIN_PASSWORD is missing in .env file!")

        user = User.query.filter_by(username=default_username).first()

        if not user:
            new_user = User(username=default_username)
            new_user.set_password(default_password)
            db.session.add(new_user)
            db.session.commit()
            print(f"Default user '{default_username}' created.")
        else:
            print(f"User '{default_username}' already exists.")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def should_send_alert(name):
    now = time.time()
    if name not in last_alert_times or now - last_alert_times[name] > ALERT_COOLDOWN:
        last_alert_times[name] = now
        return True
    return False

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    faces = Face.query.all()
    for face in faces:
        img_array = np.frombuffer(face.image, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(face.name)

def alert_saver():
    with app.app_context():
        while True:
            alert_data = alert_queue.get()
            if alert_data is None:
                break

            name, alert_type, img_bytes = alert_data

            new_alert = Alert(name=name, type=alert_type, image=img_bytes)
            db.session.add(new_alert)
            db.session.commit()

            alert_queue.task_done()

# ========== CAMERA CLASS ==========

class IPCamera:
    def __init__(self, rtsp_url, frame_width=320, frame_height=240):
        self.rtsp_url = rtsp_url
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ffmpeg_process = None
        self.start_ffmpeg_process()

    def start_ffmpeg_process(self):
        try:
            self.ffmpeg_process = subprocess.Popen(
                [
                    ffmpeg_path,
                    '-rtsp_transport', 'tcp',
                    '-i', self.rtsp_url,
                    '-vf', f'scale={self.frame_width}:{self.frame_height}',
                    '-max_delay', '1000000',
                    '-fflags', '+genpts',
                    '-buffer_size', '1000000',
                    '-pix_fmt', 'bgr24',
                    '-f', 'image2pipe',
                    '-vcodec', 'rawvideo',
                    '-'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"Error starting FFmpeg: {e}")

    def get_frame(self):
        if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
            self.start_ffmpeg_process()
        
        raw_frame = self.ffmpeg_process.stdout.read(self.frame_width * self.frame_height * 3)
        if not raw_frame:
            return None
        
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.frame_height, self.frame_width, 3)).copy()
        return frame

# ========== FRAME GENERATOR ==========

def gen_frames():
    camera = IPCamera(os.getenv('RTSP_URL'))

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            alert_type = 'familiar' if name != 'Unknown' else 'unfamiliar'

            if should_send_alert(name):
                socketio.emit('alert', {
                    'type': alert_type,
                    'message': f"{name} detected!" if name != 'Unknown' else 'Unfamiliar face detected!'
                }, namespace='/')

                face_img = frame[top:bottom, left:right]
                _, img_encoded = cv2.imencode('.jpg', face_img)
                img_bytes = img_encoded.tobytes()

                # Put alert in queue (non-blocking)
                alert_queue.put((name, alert_type, img_bytes))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ========== ROUTES ==========

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_known_face', methods=['GET', 'POST'])
@login_required
def add_known_face():
    if request.method == 'POST':
        file = request.files['image']
        name = request.form['name']
        if file and name:
            image_data = file.read()
            image_np = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                flash("Invalid image", "danger")
                return redirect(url_for('add_known_face'))

            new_face = Face(name=name, image=image_data)
            db.session.add(new_face)
            db.session.commit()

            load_known_faces()
            flash("Known face added successfully!", "success")
            return redirect(url_for('index'))
    return render_template('add_face.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user'] = user.username
            return redirect(url_for('index'))

        flash("Invalid credentials", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/alerts')
def alerts():
    # Get filter values from query string
    filter_type = request.args.get('filter', 'all')  # 'all', 'familiar', 'unknown'
    date_str = request.args.get('date', None)
    page = request.args.get('page', 1, type=int)
    per_page = 9

    # Start building query
    alerts_query = Alert.query

    # Apply type filter
    if filter_type == 'familiar':
        alerts_query = alerts_query.filter_by(type='familiar')
    elif filter_type == 'unfamiliar':
        alerts_query = alerts_query.filter_by(type='unfamiliar')

    # Apply date filter
    if date_str:
        try:
            selected_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            alerts_query = alerts_query.filter(
                db.func.date(Alert.timestamp) == selected_date
            )
        except ValueError:
            pass  # Ignore invalid date formats

    # Order by latest first
    alerts_query = alerts_query.order_by(Alert.timestamp.desc())

    # Pagination
    pagination = alerts_query.paginate(page=page, per_page=per_page, error_out=False)

    return render_template('alerts.html', alerts=pagination, filter_type=filter_type, date=date_str)


# ========== MAIN ==========

@app.template_filter('b64encode')
def b64encode_filter(data):
    return b64encode(data).decode('utf-8')

if __name__ == '__main__':
    with app.app_context():
        init_db()
        load_known_faces()

    # Start background thread for alert saving
    threading.Thread(target=alert_saver, daemon=True).start()

    socketio.run(app, debug=True)