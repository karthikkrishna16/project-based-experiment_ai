### NAME: TH KARTHIK KRISHNA

### REGISTER NO: 2122223240067

### DATE: 30/10/2025

# Classroom_Attendance
## 1.Short summary / approach:

Use a two-stage pipeline:

Face detection (find faces in an image or live stream) — e.g., MTCNN, RetinaFace, or OpenCV DNN.

Face recognition (convert faces to embeddings and match with enrolled students) — e.g., FaceNet / ArcFace embeddings + a lightweight classifier (KNN/SVM) or direct cosine similarity against stored embeddings.

Backend exposes REST APIs for:

enroll/registration (upload multiple photos per student)

mark attendance from an uploaded photo or camera snapshot

view/edit attendance

export CSV

Frontend: simple teacher dashboard (React) to register new students, trigger attendance (upload camera/photo), view / correct / export attendance.

DB: PostgreSQL (or SQLite for demo) with tables: students, embeddings, attendance_log.

You’ll provide: source code, demo video (2–3 minutes showing registration, attendance marking, dashboard), and a short report (problem, architecture diagram, dataset, accuracy).
## 2.System architecture (text diagram):
```
[Camera or Uploaded Photo]  -->  [Backend API (FastAPI / Flask)]
                                    |
                   +----------------+----------------+
                   |                                 |
            [Face Detection]                   [Student DB / Embeddings]
                   |                                 |
            [Face Crops -> Embeddings]  <--- Enrollment API saves embeddings
                   |
           [Recognition (matching)]
                   |
           [Attendance marking & store in DB]
                   |
           [Teacher Dashboard (React) <--> Backend API]
```
## 3.Tech stack (recommended)

Backend: Python, FastAPI (or Flask)

Face detection/recognition: facenet-pytorch (MTCNN + InceptionResnet face embeddings) OR insightface/ArcFace if you prefer state-of-the-art.

Image processing: OpenCV

DB: SQLite (demo) / PostgreSQL (production)

Frontend: React + Tailwind CSS (simple, responsive dashboard)

Auth: simple token-based auth for teacher (optional)

Packaging: Dockerfile for backend + frontend (optional)

Export: CSV via backend endpoint

Video demo: record with OBS or phone screen recorder
4 — Database schema (minimum)

## SQL (SQLite/Postgres-compatible):
### SQL (SQLite/Postgres-compatible):
```
CREATE TABLE students (
  id SERIAL PRIMARY KEY,
  student_id VARCHAR(64) UNIQUE NOT NULL,
  name VARCHAR(255) NOT NULL,
  roll_no VARCHAR(64),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
  embedding VECTOR, -- if using Postgres + vector extension; otherwise store as JSON/text
  source VARCHAR(255), -- filename or camera
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attendance_log (
  id SERIAL PRIMARY KEY,
  student_id INTEGER REFERENCES students(id),
  date DATE NOT NULL,
  status VARCHAR(10) CHECK (status IN ('Present','Absent','Excused')) DEFAULT 'Present',
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
If your DB doesn’t support VECTOR, store embeddings as JSON strings (TEXT) and load them into numpy arrays in backend.

## 5.Enrollment / Registration flow

Teacher opens "Register Student" form.

Upload 5–20 images per student (different angles, lighting).

Backend detects faces in uploads, crops, computes embeddings, and stores average embedding (or store multiple embeddings).

Optionally run augmentation (flip, small rotations) to improve robustness.

Key point: More images per student → better recognition.

### 6.Recognition & attendance logic (high level)

Detect faces in input photo using MTCNN / RetinaFace.

For each detected face:

Align / crop face

Compute embedding (512-d)

Compare with stored embeddings:

Option A: Compute cosine similarity with each stored student embedding; if top similarity > threshold (e.g., 0.6–0.7 depending on model), mark student as present.

Option B: Train a small classifier (SVM/KNN) on enrolled embeddings; predict class if confidence above threshold.

Mark students as Present for that date. Any enrolled student not matched = Absent.

Save attendance records in DB; provide UI for teacher to correct any mistakes.

Threshold tuning is critical — you’ll measure on hold-out images to choose threshold.

## 7.Evaluation / accuracy measurement

Prepare a dataset split: per-person enroll images vs test images (classroom photos with occlusions if possible).

Metrics:

Recognition accuracy (identification rate): fraction of correctly identified faces among detected faces.

Precision / Recall for predicted Present vs actual (if you have ground-truth labels on test images).

False Positive Rate (incorrectly marking someone present).

False Negative Rate (missed detection).

Example: evaluate embedding matching with cosine similarity on validation set and plot ROC to select threshold.

For attendance-level accuracy: compare generated attendance to ground-truth roll call on test day: compute % students correctly marked.

## 8.Important implementation details & tips

Use face alignment before feeding to embedding model.

Use batch processing for multiple faces.

For live camera feed: capture frame every N seconds (configurable) and run detection.

In crowded classroom photos (many small faces), higher-res images help.

Privacy: store embeddings (not raw images) or encrypt images if needed.

To improve: use hierarchical model — first coarse grouping (KNN) then fine SVM.

## 9.Minimum viable code skeleton

Below are key snippets to get you started quickly. Use facenet-pytorch (easy to start).

requirements.txt
```
fastapi
uvicorn[standard]
numpy
opencv-python
pillow
facenet-pytorch
sqlalchemy
psycopg2-binary   # if using Postgres
python-multipart
pandas
```
#### Backend (FastAPI) — core snippets

main.py:
```
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import io, json
from datetime import date
import sqlite3

app = FastAPI()
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Simple SQLite helper - for demo only
conn = sqlite3.connect('attendance.db', check_same_thread=False)
c = conn.cursor()
# create tables if not exist (simplified)
c.execute('''CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, student_id TEXT UNIQUE, name TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, student_id INTEGER, emb TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS attendance_log (id INTEGER PRIMARY KEY, student_id INTEGER, date TEXT, status TEXT)''')
conn.commit()

def image_from_upload(upload_file: UploadFile):
    contents = upload_file.file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    return img

def get_embedding(cropped_face: Image.Image):
    # cropped_face is PIL Image
    img_tensor = mtcnn(image=cropped_face)  # NOTE: mtcnn returns tensor when keep_all False; for single face you can preprocess differently
    # To be safe, resize and center-crop manually or use transforms to tensor; here assume cropped_face is 160x160
    img = np.array(cropped_face).astype(np.float32)
    img = Image.fromarray(img).resize((160,160))
    # Convert to tensor
    import torch
    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(t).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)

@app.post("/register")
async def register(student_id: str = Form(...), name: str = Form(...), files: list[UploadFile] = File(...)):
    # detect faces, compute embeddings, average them
    embeddings = []
    for f in files:
        img = image_from_upload(f)
        boxes, _ = mtcnn.detect(img)
        if boxes is None:
            continue
        # crop first detected face (assume one face per enrollment image)
        box = boxes[0]
        left, top, right, bottom = map(int, box)
        crop = img.crop((left, top, right, bottom))
        emb = get_embedding(crop)
        embeddings.append(emb.tolist())
    if not embeddings:
        return {"status":"no_face_detected"}
    # average embedding
    avg_emb = np.mean(np.array(embeddings), axis=0).tolist()
    # save student
    c.execute("INSERT OR IGNORE INTO students (student_id, name) VALUES (?,?)", (student_id, name))
    conn.commit()
    c.execute("SELECT id FROM students WHERE student_id=?", (student_id,))
    sid = c.fetchone()[0]
    c.execute("INSERT INTO embeddings (student_id, emb) VALUES (?,?)", (sid, json.dumps(avg_emb)))
    conn.commit()
    return {"status":"registered", "student_db_id": sid}


@app.post("/attendance_photo")
async def attendance_photo(file: UploadFile = File(...)):
    img = image_from_upload(file)
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return {"detected":0}
    detected = []
    # load all student embeddings
    c.execute("SELECT student_id, emb FROM embeddings")
    rows = c.fetchall()
    db_embs = [(r[0], np.array(json.loads(r[1]))) for r in rows]

    for box in boxes:
        left, top, right, bottom = map(int, box)
        crop = img.crop((left, top, right, bottom)).resize((160,160))
        emb = get_embedding(crop)
        # match: cosine similarity
        best = None
        best_sim = -1
        for sid, db_emb in db_embs:
            sim = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
            if sim > best_sim:
                best_sim = sim
                best = sid
        THRESH = 0.55
        if best_sim >= THRESH:
            detected.append({"student_db_id": int(best), "similarity": float(best_sim)})
            # mark attendance
            today = date.today().isoformat()
            c.execute("INSERT INTO attendance_log (student_id, date, status) VALUES (?,?,?)", (int(best), today, "Present"))
            conn.commit()
    return {"detected": len(detected), "matches": detected}
```
This is a minimal demo server. For production, you should:

add batching, async handling, better error handling, authentication,

store embeddings using a vector extension for fast nearest neighbor search (e.g., PostgreSQL + pgvector, or FAISS).

avoid re-detecting faces with MTCNN in get_embedding (use single pass).
## 10.Frontend (React) — minimal behavior

Pages:
Dashboard: select date, show attendance table with students and status, edit status inline, export CSV.
Register: form to upload images and submit student_id + name.
Mark Attendance: upload photo / connect to camera, POST to /attendance_photo.
CSV export: backend endpoint /export?date=YYYY-MM-DD that returns text/csv.

## 11.CSV export format

Header: student_id,name,roll_no,date,status,timestamp
Example:
```
student_id,name,roll_no,date,status,timestamp
S001,John Doe,12A,2025-10-30,Present,2025-10-30 09:10:23
S002,Jane Smith,12B,2025-10-30,Absent,2025-10-30 09:10:23
```
## 12.Demo script for 2–3 minute video

Intro title slide: project name + your name (3–4 seconds).

Registration (30–40s):

Open dashboard → Register Student.

Show uploading 5 images for a student → submit → show success.

Attendance marking (40–50s):

Upload a classroom photo or show live camera frame.

Show backend detecting faces (overlay boxes if UI supports) and showing matched students as Present.

Dashboard verification & edit (20–30s):

Show the teacher correcting one wrong entry (e.g., change Present → Absent).

Export attendance CSV and briefly open it.

Results & metrics (15–20s):

Show accuracy number (e.g., recognition accuracy on your test set) and short note on limitations.

Closing slide with GitHub repo link and contact.

Record clear, steady screen capture. Narrate or overlay text to explain steps.

## 13.Project report outline (1–2 pages)

Title & author

Problem statement

Objectives

System architecture (include the diagram above)

Dataset (how many images per student, augmentation)

Models used (MTCNN + FaceNet/ArcFace), hyperparameters, thresholds

Evaluation (metrics, results table)

Demo summary (what the video shows)

Limitations, future work (e.g., use FAISS, add liveness detection, handle occlusion, scale to 100s students)

How to run (setup + commands)

GitHub link

## 14.Data collection & dataset suggestions

For enrollment: collect 5–20 clear face images per student (front, left, right, occluded with glasses, different lighting).

For test: collect classroom photos with multiple students and ground-truth labels (you can annotate manually).

Public datasets (for prototyping / model validation): LFW (face verification), VGGFace2 (pretrained), WiderFace (detection evaluations). But for attendance you should use your own classroom photos.

## 15.Performance & scaling recommendations

Small class (≤50): DB + simple cosine matching is fine.

Large scale (100s+): use a vector index (FAISS, hnswlib, or Postgres + pgvector).

Real-time: only process every N-th frame and run detection on resized images.

Add liveness detection (blink detection or depth) to prevent spoofing with photos.

## 16.Checklist of deliverables you can submit

 Source code repository (backend, frontend, requirements, README with setup)

 Demo video 2–3 minutes (show registration, attendance marking, dashboard)

 Project report (1–3 pages)

 Sample dataset (a few representative enrollment images + a classroom test photo)

 Accuracy results / short evaluation (include how threshold chosen)

 Instructions to run locally (commands)

## 17.Quick "How to run (local demo)" commands

(Assuming Linux / WSL / Windows with Python)

#### 1.Create env & install:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Run server:
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open frontend (if React) at http://localhost:3000 and configure API base to http://localhost:8000.

Register students via dashboard or using curl to /register.

## 18.Privacy / ethical notes (include in report)

Get consent from students before collecting faces.

Store minimal data and secure the server.

Consider deleting raw images after computing embeddings (store only embeddings).

Inform teachers/students how data is used and retention policy.

## 19.Example evaluation result you can aim for (realistic)

Detection rate (faces detected in images): > 95% for medium-resolution classroom photos

Recognition accuracy (on hold-out test set): 85–95% depending on number/quality of enrollment images and threshold

Attendance accuracy (end-to-end vs ground truth): ~90% (with teacher corrections)

(These numbers depend heavily on dataset quality and classroom conditions.)

## 20.Final notes & next steps I can provide (pick any)

I can:

generate a complete starter repo with the exact files (backend main.py, requirements.txt, README).

produce React dashboard skeleton code (components + API calls).

prepare the short project report text (ready-to-export).

create a checklist/script for demo recording.
