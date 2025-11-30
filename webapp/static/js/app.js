/**
 * rPPG Web Application - 前端邏輯
 * - 攝像頭捕獲
 * - Canvas 繪製（影像 + ROI 框）
 * - Socket.IO 通訊
 * - Chart.js 圖表更新
 */

// ============================================
// 全局變量
// ============================================

let video, canvas, ctx;
let socket;
let isRunning = false;
let captureInterval = null;
let fpsInterval = null;
let frameCount = 0;
let lastTime = Date.now();

// Chart.js 圖表
let bvpChart, hrChart;

// ROI 顏色
const ROI_COLORS = ['#00ff00', '#0099ff', '#ff0000']; // 綠、藍、紅
const ROI_NAMES = ['Forehead', 'Left Cheek', 'Right Cheek'];

// ============================================
// 初始化
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ DOM loaded, initializing app...');

    // 獲取 DOM 元素
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetBtn = document.getElementById('resetBtn');

    // 事件監聽
    startBtn.addEventListener('click', startDetection);
    stopBtn.addEventListener('click', stopDetection);
    resetBtn.addEventListener('click', resetDetector);

    // 初始化 Socket.IO
    initSocket();

    // 初始化圖表
    initCharts();

    console.log('✅ App initialized');
});

// ============================================
// Socket.IO 通訊
// ============================================

function initSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('✅ Connected to server');
        updateStatus('已連接到服務器');
    });

    socket.on('disconnect', () => {
        console.log('❌ Disconnected from server');
        updateStatus('與服務器斷開連接');
        stopDetection();
    });

    socket.on('message', (data) => {
        console.log('📩 Message:', data.text);
        updateStatus(data.text);
    });

    socket.on('result', (data) => {
        handleResult(data);
    });

    socket.on('error', (data) => {
        console.error('⚠️ Error:', data.message);
        updateStatus(`錯誤: ${data.message}`, true);
    });
}

// ============================================
// 攝像頭控制
// ============================================

async function startDetection() {
    try {
        // 請求攝像頭訪問
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            }
        });

        video.srcObject = stream;
        video.play();

        // 等待視頻準備好
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // 開始捕獲
            isRunning = true;
            startCapture();

            // 更新按鈕狀態
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            updateStatus('檢測中...');
            console.log('✅ Detection started');
        };

    } catch (error) {
        console.error('⚠️ Camera access error:', error);
        updateStatus(`無法訪問攝像頭: ${error.message}`, true);
        alert('無法訪問攝像頭！請確保已授權瀏覽器使用攝像頭。');
    }
}

function stopDetection() {
    isRunning = false;

    // 停止捕獲
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    if (fpsInterval) {
        clearInterval(fpsInterval);
        fpsInterval = null;
    }

    // 停止攝像頭
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    // 更新按鈕狀態
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;

    updateStatus('已停止檢測');
    console.log('⏹ Detection stopped');
}

// ============================================
// 影格捕獲與發送
// ============================================

function startCapture() {
    // 每 100ms 捕獲一幀 (10 fps)
    captureInterval = setInterval(() => {
        if (!isRunning) return;

        captureAndSend();
    }, 100);

    // 計算 FPS
    fpsInterval = setInterval(updateFPS, 1000);
}

function captureAndSend() {
    // 將視頻繪製到 canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 轉換為 JPEG (quality=80)
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    // 發送給服務器
    socket.emit('frame', { image: imageData });

    frameCount++;
}

function updateFPS() {
    const now = Date.now();
    const elapsed = (now - lastTime) / 1000;
    const fps = Math.round(frameCount / elapsed);

    document.getElementById('fps').textContent = fps;

    // 重置計數器
    frameCount = 0;
    lastTime = now;
}

// ============================================
// 處理推論結果
// ============================================

function handleResult(data) {
    // 繪製 ROI 框
    drawROIs(data.face_bbox, data.roi_coords);

    // 更新 UI
    updateFrameCount(data.frame_count);
    updateStatus(data.status);

    // 更新原始 HR 到圖表（實時波形）
    if (data.hr_raw !== null) {
        updateBVPChart(data.hr_raw);
    }

    // 更新平滑後的心率顯示
    if (data.hr !== null) {
        updateHeartRate(data.hr);
    }

    // 更新心率歷史趨勢圖
    if (data.hr_history && data.hr_history.length > 0) {
        updateHRChart(data.hr_history);
    }
}

// ============================================
// Canvas 繪製
// ============================================

function drawROIs(face_bbox, roi_coords) {
    // 先繪製視頻幀
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 繪製臉部邊界框（黃色虛線）
    if (face_bbox) {
        const [x, y, w, h] = face_bbox;
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }

    // 繪製 3 個 ROI 框
    if (roi_coords && roi_coords.length === 3) {
        roi_coords.forEach((roi, index) => {
            const [x1, y1, x2, y2] = roi;
            ctx.strokeStyle = ROI_COLORS[index];
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // 繪製 ROI 標籤
            ctx.fillStyle = ROI_COLORS[index];
            ctx.font = '14px Arial';
            ctx.fillText(ROI_NAMES[index], x1 + 5, y1 - 5);
        });
    }
}

// ============================================
// UI 更新
// ============================================

function updateStatus(message, isError = false) {
    const statusText = document.getElementById('statusText');
    statusText.textContent = message;
    statusText.style.color = isError ? '#ff6b6b' : '#ffffff';
}

function updateFrameCount(count) {
    document.getElementById('frameCount').textContent = count;
}

function updateBVPBufferSize(size) {
    document.getElementById('bvpBufferSize').textContent = size;
}

function updateHeartRate(hr) {
    const hrValue = document.getElementById('hrValue');
    const hrStatus = document.getElementById('hrStatus');

    hrValue.textContent = hr.toFixed(1);

    // 狀態評估
    if (hr < 60) {
        hrStatus.textContent = '心率偏低';
        hrStatus.style.color = '#ffd700';
    } else if (hr > 100) {
        hrStatus.textContent = '心率偏高';
        hrStatus.style.color = '#ff6b6b';
    } else {
        hrStatus.textContent = '心率正常';
        hrStatus.style.color = '#4ade80';
    }
}

// ============================================
// Chart.js 圖表
// ============================================

function initCharts() {
    // BVP 波形圖
    const bvpCtx = document.getElementById('bvpChart').getContext('2d');
    bvpChart = new Chart(bvpCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'HR (即時)',
                data: [],
                borderColor: '#00ffff',
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: {
                    min: 30,
                    max: 180,
                    ticks: {
                        color: '#ffffff',
                        stepSize: 30
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            },
            animation: { duration: 0 }
        }
    });

    // 心率趨勢圖
    const hrCtx = document.getElementById('hrChart').getContext('2d');
    hrChart = new Chart(hrCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'HR (BPM)',
                data: [],
                borderColor: '#ff6b9d',
                backgroundColor: 'rgba(255, 107, 157, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: {
                    min: 40,
                    max: 140,
                    ticks: {
                        color: '#ffffff',
                        stepSize: 20
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            },
            animation: { duration: 0 }
        }
    });
}

function updateBVPChart(bvpValue) {
    const maxPoints = 100;

    // 添加新數據點
    bvpChart.data.labels.push('');
    bvpChart.data.datasets[0].data.push(bvpValue);

    // 限制顯示數量
    if (bvpChart.data.labels.length > maxPoints) {
        bvpChart.data.labels.shift();
        bvpChart.data.datasets[0].data.shift();
    }

    bvpChart.update();
}

function updateHRChart(hrHistory) {
    // 更新心率歷史圖表
    hrChart.data.labels = hrHistory.map((_, i) => i);
    hrChart.data.datasets[0].data = hrHistory;
    hrChart.update();
}

// ============================================
// 重置
// ============================================

function resetDetector() {
    if (confirm('確定要重置檢測器嗎？這將清除所有已累積的數據。')) {
        socket.emit('reset');

        // 重置圖表
        bvpChart.data.labels = [];
        bvpChart.data.datasets[0].data = [];
        bvpChart.update();

        hrChart.data.labels = [];
        hrChart.data.datasets[0].data = [];
        hrChart.update();

        // 重置 UI
        document.getElementById('hrValue').textContent = '--';
        document.getElementById('hrStatus').textContent = '等待數據...';
        document.getElementById('frameCount').textContent = '0';
        document.getElementById('bvpBufferSize').textContent = '0';

        console.log('✅ Detector reset requested');
    }
}
