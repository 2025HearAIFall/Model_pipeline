# inference.py (GRU 호환 v3 버전)
# -----------------------------------------------------------------------------
# [변경 사항]
# 1. 모델 경로 v3로 변경
# 2. onnx_predict 함수: Cell state 제거 (GRU 호환)
# -----------------------------------------------------------------------------

import os
import json
import pickle
import time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import onnxruntime as ort
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 0. 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# [모델 A 경로 수정] v3 (Bi-GRU) 파일명으로 변경
ENCODER_PATH_A = os.path.join(BASE_DIR, 'model_a_v3_encoder.onnx')
DECODER_PATH_A = os.path.join(BASE_DIR, 'model_a_v3_decoder.onnx')
VOCAB_PATH_A   = os.path.join(BASE_DIR, 'vocab.pkl')

# (모델 B)
MODEL_B_PATH_OR_NAME = "." # 혹은 huggingface 모델명

# (모델 C: 감정)
MODEL_C_ONNX_PATH = os.path.join(BASE_DIR, 'cnn_gru_attn_10mb.onnx')
MODEL_C_PT_PATH   = os.path.join(BASE_DIR, 'cnn_gru_attn_10mb.pt')
EMOTION_CLASSES = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
IDX2EMO = {i: c for i, c in enumerate(EMOTION_CLASSES)}

# --- 유틸 ---
def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z, dtype=np.float32)
    return e / e.sum()

# -----------------------------------------------------------------------------
# 1. Vocab 클래스 (Pickle 로딩용)
# -----------------------------------------------------------------------------
class Vocabulary:
    def __init__(self, tokenizer=None, min_freq=2):
        self.tokenizer = tokenizer
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def __len__(self): return len(self.itos)

def simple_tokenizer(text): return text.split(' ')

# -----------------------------------------------------------------------------
# 2. 모델 A (수어) 로드
# -----------------------------------------------------------------------------
print("🔄 모델 A (수어) 로딩 중...")
try:
    with open(VOCAB_PATH_A, 'rb') as f:
        vocab = pickle.load(f)
    
    encoder_session = ort.InferenceSession(ENCODER_PATH_A, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(DECODER_PATH_A, providers=['CPUExecutionProvider'])
    print("✅ 모델 A (ONNX GRU) 로드 완료.")
except Exception as e:
    print(f"❌ 모델 A 로드 실패: {e}")
    vocab, encoder_session, decoder_session = None, None, None

# 🔥 [핵심 수정] GRU용 예측 함수 (Cell State 제거)
def onnx_predict(encoder_sess, decoder_sess, src_seq_np, max_output_len, sos_idx, eos_idx):
    try:
        # 1. Encoder 실행
        encoder_inputs = {'input_keypoints': src_seq_np}
        # GRU Encoder는 hidden만 반환 (cell 없음)
        encoder_outputs, hidden = encoder_sess.run(None, encoder_inputs)
        
        # 2. Decoder 준비
        trg_input = np.array([sos_idx], dtype=np.int64) # (1,)
        output_tokens = []
        
        for _ in range(max_output_len):
            decoder_inputs = {
                'input_token': trg_input,
                'in_hidden': hidden,
                'encoder_outputs': encoder_outputs
            }
            
            # GRU Decoder는 hidden만 반환
            logits, hidden = decoder_sess.run(None, decoder_inputs)
            
            # 다음 토큰 예측
            top1_item = int(np.argmax(logits, axis=1)[0])
            
            if top1_item == eos_idx:
                break
                
            output_tokens.append(top1_item)
            trg_input = np.array([top1_item], dtype=np.int64)
            
        return output_tokens
        
    except Exception as e:
        print(f"⚠️ 예측 중 에러: {e}")
        return []

# -----------------------------------------------------------------------------
# 3. 모델 B (문맥 복원) 로드
# -----------------------------------------------------------------------------
print("🔄 모델 B (문맥) 로딩 중...")
try:
    gec_tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH_OR_NAME)
    gec_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_B_PATH_OR_NAME).to(DEVICE)
    print("✅ 모델 B 로드 완료.")
except:
    print("⚠️ 모델 B 로드 실패 (건너뜀)")
    gec_model, gec_tokenizer = None, None

# -----------------------------------------------------------------------------
# 4. 모델 C (감정) 로드
# -----------------------------------------------------------------------------
# ... (사용자님의 기존 감정 모델 클래스 코드 유지 - 길이 관계상 핵심만 포함) ...
class EmotionInfer:
    def __init__(self):
        self.sess = None
        if os.path.exists(MODEL_C_ONNX_PATH):
            self.sess = ort.InferenceSession(MODEL_C_ONNX_PATH, providers=['CPUExecutionProvider'])
            self.in_name = self.sess.get_inputs()[0].name
            self.out_name = self.sess.get_outputs()[0].name
            print("✅ 모델 C (감정 ONNX) 로드 완료.")
        else:
            print("⚠️ 모델 C 없음.")

    def infer_from_file(self, audio_path):
        if not self.sess: return "Unknown", 0.0, None
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            # Padding/Crop to 128
            if mfcc.shape[1] < 128:
                pad = np.zeros((40, 128 - mfcc.shape[1]))
                mfcc = np.hstack((mfcc, pad))
            else:
                mfcc = mfcc[:, :128]
            
            x = mfcc[None, None, :, :].astype(np.float32)
            logits = self.sess.run([self.out_name], {self.in_name: x})[0][0]
            probs = softmax_np(logits)
            idx = int(probs.argmax())
            return IDX2EMO[idx], float(probs[idx]), probs
        except Exception as e:
            print(f"Emo Error: {e}")
            return "Error", 0.0, None

emo = EmotionInfer()

# -----------------------------------------------------------------------------
# 5. 모델 D (STT) 로드
# -----------------------------------------------------------------------------
print("🔄 모델 D (STT) 로딩 중...")
try:
    stt_model = whisper.load_model("base", device=DEVICE)
    print("✅ 모델 D 로드 완료.")
except:
    print("⚠️ 모델 D 로드 실패.")
    stt_model = None