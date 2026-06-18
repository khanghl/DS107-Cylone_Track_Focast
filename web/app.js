const API_BASE_URL = 'http://127.0.0.1:8000/api';

// UI Elements
const stormSelect = document.getElementById('stormSelect');
const timeSelect = document.getElementById('timeSelect');
const predictBtn = document.getElementById('predictBtn');
const infoDPE = document.getElementById('infoDPE');
const infoLat = document.getElementById('infoLat');
const infoLon = document.getElementById('infoLon');

// Global variables
let map, windChart;
let currentTrackLayer = L.layerGroup();
let predictionLayer = L.layerGroup();
let stormTimes = [];
let allStorms = [];

// ==========================================
// 1. INIT LEAFLET MAP
// ==========================================
function initMap() {
    map = L.map('map').setView([15.0, 115.0], 5);

    // Sử dụng Google Maps bản tiếng Việt
    L.tileLayer('http://mt0.google.com/vt/lyrs=m&hl=vi&x={x}&y={y}&z={z}', {
        attribution: '&copy; Google Maps',
        maxZoom: 20
    }).addTo(map);

    currentTrackLayer.addTo(map);
    predictionLayer.addTo(map);
}

// ==========================================
// 2. INIT CHART.JS
// ==========================================
function initChart() {
    const ctx = document.getElementById('windChart').getContext('2d');
    
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";

    windChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Hiện tại', '+6h', '+12h', '+24h'],
            datasets: [
                {
                    label: 'Gió Thực Tế (Observed)',
                    data: [null, null, null, null],
                    borderColor: '#1d4ed8',
                    backgroundColor: '#1d4ed8',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 4
                },
                {
                    label: 'Gió Dự Báo (Predicted)',
                    data: [null, null, null, null],
                    borderColor: '#38bdf8',
                    backgroundColor: '#38bdf8',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.3,
                    pointRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12 }
                }
            },
            scales: {
                y: {
                    title: { display: true, text: 'Tốc độ (knots)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });
}

// ==========================================
// 3. BACKGROUND WIND ANIMATION
// ==========================================
function initWindCanvas() {
    const canvas = document.getElementById('windCanvas');
    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    class Particle {
        constructor() {
            this.reset();
        }
        reset() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            // Gió thổi từ đông sang tây (hơi chếch)
            this.vx = -(Math.random() * 2 + 1);
            this.vy = (Math.random() - 0.5) * 0.5;
            this.length = Math.random() * 50 + 20;
            this.alpha = Math.random() * 0.5 + 0.1;
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < -this.length || this.y < 0 || this.y > height) {
                this.x = width + this.length;
                this.y = Math.random() * height;
            }
        }
        draw() {
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(this.x + this.length, this.y - this.vy * 10);
            ctx.strokeStyle = `rgba(56, 189, 248, ${this.alpha})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    for (let i = 0; i < 150; i++) particles.push(new Particle());

    function animate() {
        ctx.clearRect(0, 0, width, height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        requestAnimationFrame(animate);
    }
    animate();
}

// ==========================================
// 4. API CALLS & LOGIC
// ==========================================
async function loadStorms() {
    try {
        const res = await fetch(`${API_BASE_URL}/storms`);
        if (!res.ok) throw new Error("Failed to fetch storms");
        allStorms = await res.json();
        
        stormSelect.innerHTML = '<option value="">-- Chọn Cơn Bão --</option>';
        allStorms.forEach(s => {
            // Hiển thị tên nếu có, nếu không thì hiện SID
            stormSelect.innerHTML += `<option value="${s.sid}">${s.sid} (Năm ${s.year})</option>`;
        });
    } catch (err) {
        console.error(err);
        stormSelect.innerHTML = '<option value="">Lỗi kết nối API!</option>';
    }
}

async function loadStormTimes(sid) {
    try {
        timeSelect.innerHTML = '<option value="">Đang tải...</option>';
        timeSelect.disabled = true;
        predictBtn.disabled = true;

        const res = await fetch(`${API_BASE_URL}/storms/${sid}/times`);
        if (!res.ok) throw new Error("Failed to fetch times");
        stormTimes = await res.json();

        timeSelect.innerHTML = '<option value="">-- Chọn mốc thời gian --</option>';
        stormTimes.forEach((t, i) => {
            timeSelect.innerHTML += `<option value="${i}">${t.time_str}</option>`;
        });
        
        timeSelect.disabled = false;
        
        // Show selected storm info
        const selectedStorm = allStorms.find(s => s.sid === sid);
        if (selectedStorm) {
            infoDPE.textContent = selectedStorm.dpe_24h;
        }

    } catch (err) {
        console.error(err);
        timeSelect.innerHTML = '<option value="">Lỗi kết nối API!</option>';
    }
}

async function doPrediction() {
    const selectedIdx = timeSelect.value;
    if (selectedIdx === "") return;
    
    const timeData = stormTimes[selectedIdx];
    
    try {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span>Đang xử lý...</span>';

        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: timeData.index })
        });
        
        if (!res.ok) throw new Error("Prediction failed");
        const data = await res.json();
        
        updateUI(data, selectedIdx);

    } catch (err) {
        console.error(err);
        alert("Lỗi khi lấy dữ liệu dự báo. Xem console.");
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<span>Dự Báo Quỹ Đạo & Cường Độ</span>';
    }
}

function updateUI(data, currentIdx) {
    // 1. Draw Map
    currentTrackLayer.clearLayers();
    predictionLayer.clearLayers();

    // Lấy tọa độ từ quá khứ đến hiện tại
    const latlngs = [];
    for(let i=0; i<=currentIdx; i++) {
        latlngs.push([stormTimes[i].lat, stormTimes[i].lon]);
    }
    
    // Hàm tính khoảng cách
    function getDistanceInMeters(lat1, lon1, lat2, lon2) {
        const R = 6371e3;
        const p1 = lat1 * Math.PI/180, p2 = lat2 * Math.PI/180;
        const dp = (lat2-lat1) * Math.PI/180;
        const dl = (lon2-lon1) * Math.PI/180;
        const a = Math.sin(dp/2) * Math.sin(dp/2) +
                Math.cos(p1) * Math.cos(p2) *
                Math.sin(dl/2) * Math.sin(dl/2);
        return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    }

    // Khởi tạo polyline rỗng
    const animPolyline = L.polyline([], {color: '#1d4ed8', weight: 2}).addTo(currentTrackLayer);
    
    // Icon bão xoay xoay
    const stormIcon = L.divIcon({
        className: 'custom-storm-icon',
        html: '<div class="cyclone-spin">🌀</div>',
        iconSize: [20, 20],
        iconAnchor: [0, 0]
    });

    // Khởi tạo marker chạy
    const currentMarker = L.marker(latlngs[0], {icon: stormIcon}).addTo(currentTrackLayer);

    const p6 = data.predictions["6h"];
    const p12 = data.predictions["12h"];
    const p24 = data.predictions["24h"];

    // Bắt đầu từ điểm đầu tiên với độ thu phóng gần
    map.setView(latlngs[0], 6);

    // Hiệu ứng animation di chuyển từ quá khứ
    let step = 0;
    const animInterval = setInterval(() => {
        if (step >= latlngs.length) {
            clearInterval(animInterval);
            currentMarker.bindPopup("Current Position").openPopup();
            
            // Phóng to để bao quát toàn bộ hành trình (quá khứ + tương lai)
            const allBounds = L.latLngBounds(latlngs);
            allBounds.extend([p24.pred_lat, p24.pred_lon]);
            allBounds.extend([p24.true_lat, p24.true_lon]);
            map.flyToBounds(allBounds.pad(0.3), {duration: 1.5});

            // Sau khi vẽ xong quá khứ và zoom xong, vẽ các điểm dự báo
            setTimeout(() => {
                const baseLat = data.current_lat;
                const baseLon = data.current_lon;

                // Vẽ quỹ đạo thực tế tương lai (Màu đen)
                L.polyline([[baseLat, baseLon], [p6.true_lat, p6.true_lon], [p12.true_lat, p12.true_lon], [p24.true_lat, p24.true_lon]], {color: '#000000', weight: 2}).addTo(predictionLayer);
                L.circleMarker([p6.true_lat, p6.true_lon], {color: '#000000', fillColor: '#000', fillOpacity: 1, radius: 4}).addTo(predictionLayer);
                L.circleMarker([p12.true_lat, p12.true_lon], {color: '#000000', fillColor: '#000', fillOpacity: 1, radius: 4}).addTo(predictionLayer);
                L.circleMarker([p24.true_lat, p24.true_lon], {color: '#000000', fillColor: '#000', fillOpacity: 1, radius: 4}).addTo(predictionLayer);

                // 6h
                L.polyline([[baseLat, baseLon], [p6.pred_lat, p6.pred_lon]], {color: '#facc15', weight: 2, dashArray: '5, 5'}).addTo(predictionLayer);
                L.circleMarker([p6.pred_lat, p6.pred_lon], {color: '#facc15', radius: 5}).addTo(predictionLayer);
                const dpe6 = getDistanceInMeters(p6.pred_lat, p6.pred_lon, p6.true_lat, p6.true_lon);
                L.circle([p6.pred_lat, p6.pred_lon], {radius: dpe6, color: '#000', weight: 1, dashArray: '4,4', fillColor: '#000', fillOpacity: 0.1}).addTo(predictionLayer);
                
                setTimeout(() => {
                    // 12h
                    L.polyline([[p6.pred_lat, p6.pred_lon], [p12.pred_lat, p12.pred_lon]], {color: '#f97316', weight: 2, dashArray: '5, 5'}).addTo(predictionLayer);
                    L.circleMarker([p12.pred_lat, p12.pred_lon], {color: '#f97316', radius: 5}).addTo(predictionLayer);
                    const dpe12 = getDistanceInMeters(p12.pred_lat, p12.pred_lon, p12.true_lat, p12.true_lon);
                    L.circle([p12.pred_lat, p12.pred_lon], {radius: dpe12, color: '#000', weight: 1, dashArray: '4,4', fillColor: '#000', fillOpacity: 0.1}).addTo(predictionLayer);

                    setTimeout(() => {
                        // 24h
                        L.polyline([[p12.pred_lat, p12.pred_lon], [p24.pred_lat, p24.pred_lon]], {color: '#ef4444', weight: 2, dashArray: '5, 5'}).addTo(predictionLayer);
                        L.circleMarker([p24.pred_lat, p24.pred_lon], {color: '#ef4444', radius: 5}).addTo(predictionLayer);
                        const dpe24 = getDistanceInMeters(p24.pred_lat, p24.pred_lon, p24.true_lat, p24.true_lon);
                        L.circle([p24.pred_lat, p24.pred_lon], {radius: dpe24, color: '#000', weight: 1, dashArray: '4,4', fillColor: '#000', fillOpacity: 0.1}).addTo(predictionLayer);
                    }, 300);
                }, 300);
            }, 1500); // Chờ flyToBounds hoàn tất (1.5s)

            return;
        }
        
        animPolyline.addLatLng(latlngs[step]);
        currentMarker.setLatLng(latlngs[step]);
        
        // Map pan theo bão
        map.panTo(latlngs[step], {animate: true, duration: 0.15});
        
        step++;
    }, 200); // Tốc độ di chuyển: 200ms mỗi điểm

    // Update Info Panel
    infoLat.textContent = data.current_lat.toFixed(2);
    infoLon.textContent = data.current_lon.toFixed(2);

    // 2. Update Chart
    const currentWind = data.current_wind;
    
    windChart.data.datasets[0].data = [
        currentWind,
        p6.true_wind,
        p12.true_wind,
        p24.true_wind
    ];
    
    windChart.data.datasets[1].data = [
        currentWind, // Bắt đầu cùng một điểm
        p6.pred_wind,
        p12.pred_wind,
        p24.pred_wind
    ];
    
    windChart.update();
}

// ==========================================
// EVENTS
// ==========================================
stormSelect.addEventListener('change', (e) => {
    if (e.target.value) {
        loadStormTimes(e.target.value);
    } else {
        timeSelect.innerHTML = '<option value="">Vui lòng chọn bão trước</option>';
        timeSelect.disabled = true;
        predictBtn.disabled = true;
    }
});

timeSelect.addEventListener('change', (e) => {
    predictBtn.disabled = !e.target.value;
});

predictBtn.addEventListener('click', doPrediction);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initChart();
    initWindCanvas();
    loadStorms();
});
