# ======================================================
# Hand Bridge Socket.IO Server (Full Code)
# ======================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64, io, os, torch, traceback, cv2, re
from PIL import Image
import numpy as np
import builtins
import mediapipe as mp

# ------------------------------------------------------
# 1. Path & Vocab Setup
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

class Vocabulary:
    def __init__(self, tokenizer=None, min_freq=2):
        self.tokenizer = tokenizer; self.itos = {}; self.stoi = {}
        self.min_freq = min_freq; self.pad_idx = 0; self.sos_idx = 1; self.eos_idx = 2
    def __len__(self): return len(self.itos)

def simple_tokenizer(text): return text.split(' ')

builtins.Vocabulary = Vocabulary
builtins.simple_tokenizer = simple_tokenizer

# ------------------------------------------------------
# 2. Load Models
# ------------------------------------------------------
# inference.py가 같은 폴더에 있어야 합니다.
from inference import (
    vocab, encoder_session, decoder_session,          
    gec_model, gec_tokenizer, stt_model, emo,         
    onnx_predict                                      
)

# ------------------------------------------------------
# 3. MediaPipe Setup
# ------------------------------------------------------
mp_holistic = mp.solutions.holistic

try:
    holistic_processor = mp_holistic.Holistic(
        static_image_mode=True, 
        model_complexity=1,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    print("✅ [app] MediaPipe loaded (Static Mode)")
except Exception as e:
    holistic_processor = None
    print(f"❌ [app] MediaPipe failed: {e}")

# ------------------------------------------------------
# 4. Helper Functions (여기가 누락되어 에러가 났던 부분입니다)
# ------------------------------------------------------
def _extract_kps(frame_bgr, holistic):
    """MediaPipe를 사용하여 랜드마크(Keypoints) 추출"""
    if not holistic: return np.zeros(150, dtype=np.float32)
    
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    res = holistic.process(img)
    
    # Pose (33개 중 필요한 것만 사용하거나 전체 사용) - 여기선 예시로 66개(33*2) 가정
    pose = np.zeros(66, dtype=np.float32)
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark): 
            pose[i*2], pose[i*2+1] = lm.x, lm.y
            
    # Left Hand (21*2 = 42)
    lh = np.zeros(42, dtype=np.float32)
    if res.left_hand_landmarks:
        for i, lm in enumerate(res.left_hand_landmarks.landmark): 
            lh[i*2], lh[i*2+1] = lm.x, lm.y
            
    # Right Hand (21*2 = 42)
    rh = np.zeros(42, dtype=np.float32)
    if res.right_hand_landmarks:
        for i, lm in enumerate(res.right_hand_landmarks.landmark): 
            rh[i*2], rh[i*2+1] = lm.x, lm.y
        
    # 총 150차원 (Pose 66 + LH 42 + RH 42 = 150)
    kps = np.concatenate([pose, lh, rh])
    
    # 감지된 게 거의 없으면 0으로 리턴
    if np.sum(np.abs(kps)) < 0.01: return np.zeros(150, dtype=np.float32)
    return kps

def _resample(buffer, target_len=30):
    """프레임 수를 모델 입력 크기에 맞게 조절"""
    arr = np.array(buffer, dtype=np.float32)
    if len(arr) == 0: return np.zeros((target_len, 150), dtype=np.float32)
    indices = np.linspace(0, len(arr)-1, target_len, dtype=int)
    return arr[indices]

def _prepare(arr):
    """모델 입력 전처리 (Delta 등)"""
    mot = np.zeros_like(arr)
    if len(arr) > 1: mot[1:] = arr[1:] - arr[:-1]
    return np.expand_dims(np.concatenate([arr, mot], axis=1), axis=0)

# ------------------------------------------------------
# 5. Server Config (SocketIO)
# ------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

# async_mode='eventlet' 권장 (pip install eventlet 필요)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FRAMES = 50  

# 사용자별 데이터 관리를 위한 딕셔너리
# 구조: { 'session_id': { 'buffer': [], 'room': 'room_id' } }
users = {}

# ------------------------------------------------------
# 6. Socket Events
# ------------------------------------------------------

@socketio.on('join')
def on_join(data):
    room = data.get('room', 'default')
    join_room(room)
    users[request.sid] = {'buffer': [], 'room': room}
    print(f"✅ User {request.sid} joined room: {room}")
    emit('system_msg', {'msg': f"새로운 사용자가 입장했습니다."}, to=room)

@socketio.on('disconnect')
def on_disconnect():
    if request.sid in users:
        room = users[request.sid]['room']
        del users[request.sid]
        print(f"❌ User {request.sid} disconnected")
        emit('system_msg', {'msg': "사용자가 퇴장했습니다."}, to=room)

@socketio.on('sign_data')
def handle_sign_data(data):
    # 클라이언트가 보낸 프레임 처리
    sid = request.sid
    if sid not in users: return

    try:
        # Base64 이미지 디코딩
        f_b64 = data['frame'].split(",")[1] if "," in data['frame'] else data['frame']
        img = Image.open(io.BytesIO(base64.b64decode(f_b64))).convert("RGB")
        
        # 여기서 _extract_kps 함수를 호출합니다 (위에서 정의했으므로 에러 안 남)
        kps = _extract_kps(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), holistic_processor)
        
        users[sid]['buffer'].append(kps)
        curr_len = len(users[sid]['buffer'])
        progress = int((curr_len / TARGET_FRAMES) * 100)

        # 진행률 업데이트는 나에게만
        emit('progress_update', {'progress': progress}, to=sid)

        if curr_len >= TARGET_FRAMES:
            # 50프레임 도달 -> 분석 시작
            print(f"🚀 Analyzing Sign for {sid}...")
            buffer = users[sid]['buffer']
            users[sid]['buffer'] = [] # 버퍼 초기화

            resampled = _resample(buffer)
            inp = _prepare(resampled)

            # 모델 추론
            raw_text = "..."
            if encoder_session and vocab:
                sos_id = vocab.stoi.get("<SOS>", 1)
                eos_id = vocab.stoi.get("<EOS>", 2)
                pred_idx = onnx_predict(encoder_session, decoder_session, inp, 50, sos_id, eos_id)
                tokens = [vocab.itos.get(i, "") for i in pred_idx]
                raw_text = " ".join([t for t in tokens if t not in ["<SOS>", "<PAD>", "<EOS>"]]).strip()

            corrected = raw_text
            if gec_model and raw_text:
                try:
                    inp_g = gec_tokenizer(raw_text, return_tensors="pt").to(DEVICE)
                    out_g = gec_model.generate(**inp_g, max_length=50)
                    corrected = gec_tokenizer.decode(out_g[0], skip_special_tokens=True)
                except: pass
            
            # 결과 방송 (같은 방 사람들에게 모두 전송)
            room = users[sid]['room']
            emit('chat_message', {
                'type': 'sign',
                'text': raw_text,
                'corrected': corrected,
                'sender': sid
            }, to=room)

    except Exception as e:
        print(f"Sign Error: {e}")
        traceback.print_exc()

@socketio.on('voice_data')
def handle_voice_data(data):
    sid = request.sid
    if sid not in users: return
    room = users[sid]['room']

    try:
        # 오디오 바이너리 저장
        audio_data = data['audio']
        # 확장자를 webm으로 저장 (브라우저 MediaRecorder 기본 포맷)
        filename = f"temp_{sid}.webm"
        save_path = os.path.join(BASE_DIR, filename)
        
        with open(save_path, "wb") as f:
            f.write(audio_data)
            
        # STT & 감정 분석
        rec_text = ""
        emotion = "neutral"
        
        if stt_model:
            # webm 파일도 ffmpeg가 설치되어 있다면 Whisper가 처리 가능
            res = stt_model.transcribe(save_path, language="ko")
            rec_text = res.get("text", "").strip()
            
        if emo:
            try:
                emotion, conf, _ = emo.infer_from_file(save_path)
            except: pass
            
        if os.path.exists(save_path): os.remove(save_path)

        # 결과 방송
        emit('chat_message', {
            'type': 'voice',
            'text': rec_text,
            'emotion': emotion,
            'sender': sid
        }, to=room)

    except Exception as e:
        print(f"Voice Error: {e}")
        traceback.print_exc()

# ------------------------------------------------------
# 7. Routes (HTML/Static Files)
# ------------------------------------------------------
@app.route('/')
def serve_index(): return send_from_directory(FRONTEND_DIR, 'index.html')
@app.route('/demo.html')
def serve_demo(): return send_from_directory(FRONTEND_DIR, 'demo.html')
@app.route('/assets/<path:filename>')
def serve_assets(filename): return send_from_directory(os.path.join(FRONTEND_DIR, 'assets'), filename)

if __name__ == "__main__":
    print(f"\n🚀 Socket.IO Server running on port 8000")
    socketio.run(app, host="0.0.0.0", port=8000)