// script.js
const socket = io("http://10.117.121.37:8000");
const video = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const btnSign = document.getElementById('btn-sign');
const btnVoice = document.getElementById('btn-voice');
const btnJoin = document.getElementById('btn-join');
const roomIdInput = document.getElementById('roomIdInput');

const chatBox = document.getElementById('chat-box');
const progressBar = document.getElementById('progressBar');
const statusBadge = document.getElementById('status-badge');

let intervalId = null;
let mediaRecorder = null;
let currentRoom = "";

// 0. 초기 설정 (카메라)
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => alert("카메라 권한 필요"));

// 1. 방 입장
function joinRoom() {
    const room = roomIdInput.value;
    if (!room) return alert("방 번호를 입력하세요");
    
    currentRoom = room;
    socket.emit('join', { room: room });
    
    // 버튼 활성화
    btnJoin.disabled = true;
    btnJoin.innerText = "접속됨";
    btnSign.disabled = false;
    btnVoice.disabled = false;
    
    addSystemMessage(`방 [${room}]에 입장했습니다.`);
}

// 2. 소켓 이벤트 수신 (서버 -> 클라이언트)
socket.on('system_msg', (data) => {
    addSystemMessage(data.msg);
});

socket.on('progress_update', (data) => {
    // 수어 녹화 진행률
    const pct = data.progress;
    progressBar.style.width = pct + "%";
    statusBadge.innerText = `🔴 녹화 중 (${pct}%)`;
    if (pct >= 100) stopRecordingUI();
});

socket.on('chat_message', (data) => {
    // 서버에서 분석 완료된 메시지가 오면 화면에 표시
    // data.sender가 socket.id와 같으면 '나', 다르면 '상대방'
    const isMe = (data.sender === socket.id);
    
    if (data.type === 'sign') {
        addMessage('sign', data.translation, data.corrected, isMe);
        if (!isMe) speakText(data.corrected || data.translation); // 상대방 말만 읽어주기
    } else {
        const text = `${data.text} <small>(${data.emotion})</small>`;
        addMessage('voice', text, null, isMe);
    }
});

// 3. 수어 녹화 (소켓 전송)
btnSign.addEventListener('click', () => {
    if (intervalId) return;
    btnSign.disabled = true;
    btnVoice.disabled = true;
    statusBadge.innerText = "🔴 준비 중...";
    statusBadge.style.color = "#ff4b4b";

    intervalId = setInterval(() => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg', 0.5);
        // fetch 대신 emit 사용 (훨씬 빠름)
        socket.emit('sign_data', { frame: frameData });
    }, 100);
});

function stopRecordingUI() {
    if (intervalId) { clearInterval(intervalId); intervalId = null; }
    btnSign.disabled = false;
    btnVoice.disabled = false;
    statusBadge.innerText = "대기 중";
    statusBadge.style.color = "white";
    progressBar.style.width = "0%";
}

// 4. 음성 녹음 (소켓 전송)
btnVoice.addEventListener('click', async () => {
    if (intervalId) return;
    
    btnSign.disabled = true;
    btnVoice.disabled = true;
    statusBadge.innerText = "🎤 준비...";
    statusBadge.style.color = "#ffeb33";
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    let chunks = [];

    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        
        // 바이너리 데이터 전송
        socket.emit('voice_data', { audio: blob });
        
        stopRecordingUI();
        statusBadge.innerText = "분석 중...";
    };

    mediaRecorder.start();
    
    // 5초 타이머 (UI용)
    let progress = 0;
    intervalId = setInterval(() => {
        progress += 2;
        progressBar.style.width = `${progress}%`;
        statusBadge.innerText = `🎤 녹음 (${progress}%)`;
        if (progress >= 100) {
            clearInterval(intervalId);
            intervalId = null;
            mediaRecorder.stop();
        }
    }, 100);
});

// UI 헬퍼 함수들
function addSystemMessage(text) {
    const div = document.createElement('div');
    div.className = 'system-msg';
    div.innerText = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addMessage(type, text, subText, isMe) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    
    // 내 메시지면 오른쪽, 상대방이면 왼쪽
    if (isMe) msgDiv.classList.add('msg-me'); // CSS 추가 필요
    else msgDiv.classList.add(type === 'sign' ? 'msg-sign' : 'msg-voice');

    let html = "";
    if (type === 'sign') {
        html = `<span class="name">${isMe ? '나 (수어)' : '상대 (수어)'}</span>${text}`;
        if (subText && subText !== text && subText !== "...") {
            html += `<span class="correction-text">🧩 ${subText}</span>`;
        }
    } else {
         html = `<span class="name">${isMe ? '나' : '상대'}</span>${text}`;
    }
    msgDiv.innerHTML = html;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function speakText(text) {
    if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = 'ko-KR';
        window.speechSynthesis.speak(utter);
    }
}