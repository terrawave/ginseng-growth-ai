/**
 * 새싹인삼 생육 인식 시스템 - 컴팩트 대시보드
 */

class GinsengApp {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.autoControl = false;
        this.currentStage = 'unknown';
        this.targets = null;
        this.detectionCount = 0;

        // 펌프 상태 추적 (DOM 대신 변수로 관리)
        this.pumpStates = {
            'nutrient_ab': false,
            'nutrient_c': false,
            'mixer': false
        };

        this.init();
    }

    init() {
        this.connectWebSocket();
        this.bindEvents();
        console.log('[App] 초기화 완료');
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('[WS] 연결됨');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        };

        this.ws.onclose = () => {
            console.log('[WS] 연결 해제');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('[WS] 오류:', error);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (e) {
                console.error('[WS] 파싱 오류:', e);
            }
        };
    }

    handleMessage(data) {
        switch (data.type) {
            case 'init':
                this.handleInit(data);
                break;

            case 'sensor_data':
                this.updateSensorData(data.sensor);
                this.updateControllerStatus(data.controller);
                if (data.detection) {
                    this.updateDetectionStatus(data.detection);
                }
                break;

            case 'detection':
                this.updateDetection(data);
                break;

            case 'pump_action':
                if (data.pump === 'nutrient_ab') {
                    this.updatePumpButton('pump-ab-btn', data.state);
                } else if (data.pump === 'nutrient_c') {
                    this.updatePumpButton('pump-c-btn', data.state);
                } else if (data.pump === 'mixer') {
                    this.updatePumpButton('pump-mixer-btn', data.state);
                }
                const pumpNames = { 'nutrient_ab': '양액AB', 'nutrient_c': '양액C', 'mixer': '교반기' };
                this.addLog(`[펌프] ${pumpNames[data.pump] || data.pump} = ${data.state ? 'ON' : 'OFF'}`, 'action');
                break;
        }
    }

    handleInit(data) {
        console.log('[App] 초기 데이터 수신:', data);

        if (data.controller) {
            this.autoControl = data.controller.is_enabled;
            document.getElementById('auto-toggle').checked = this.autoControl;
            this.updateAutoStatus();
        }

        if (data.detector) {
            this.updateDetectionStatus(data.detector);
        }

        this.addLog('시스템 연결됨');
    }

    updateConnectionStatus(connected) {
        const status = document.getElementById('connection-status');
        if (connected) {
            status.textContent = '연결됨';
            status.className = 'indicator connected';
        } else {
            status.textContent = '연결중...';
            status.className = 'indicator disconnected';
        }
    }

    updateAutoStatus() {
        const status = document.getElementById('auto-status');
        status.textContent = `자동: ${this.autoControl ? 'ON' : 'OFF'}`;
    }

    updateSensorData(sensor) {
        if (!sensor) return;

        // EC
        const ecValue = sensor.ec?.toFixed(2) || '-';
        document.getElementById('ec-value').textContent = ecValue;

        // pH
        const phValue = sensor.ph?.toFixed(2) || '-';
        document.getElementById('ph-value').textContent = phValue;

        // 수온
        const tempValue = sensor.water_temp?.toFixed(1) || '-';
        document.getElementById('temp-value').textContent = tempValue;

        // 목표값 바 업데이트
        if (this.targets) {
            this.updateSensorBars(sensor);
        }
    }

    updateSensorBars(sensor) {
        const t = this.targets;

        // EC 바
        const ecTargetBar = document.getElementById('ec-target-bar');
        const ecCurrentBar = document.getElementById('ec-current-bar');
        const ecTargetText = document.getElementById('ec-target');
        const ecStatus = document.getElementById('ec-status');

        ecTargetText.textContent = `목표: ${t.ec_min} - ${t.ec_max}`;

        // 바 위치 계산 (0-3 mS/cm 범위 기준)
        const ecMin = 0, ecMax = 3;
        const targetLeft = ((t.ec_min - ecMin) / (ecMax - ecMin)) * 100;
        const targetWidth = ((t.ec_max - t.ec_min) / (ecMax - ecMin)) * 100;
        const currentPos = ((sensor.ec - ecMin) / (ecMax - ecMin)) * 100;

        ecTargetBar.style.left = `${targetLeft}%`;
        ecTargetBar.style.width = `${targetWidth}%`;
        ecCurrentBar.style.left = `${Math.min(Math.max(currentPos, 0), 100)}%`;

        // 상태 표시
        if (sensor.ec < t.ec_min) {
            ecStatus.textContent = '낮음';
            ecStatus.className = 'sensor-status low';
        } else if (sensor.ec > t.ec_max) {
            ecStatus.textContent = '높음';
            ecStatus.className = 'sensor-status high';
        } else {
            ecStatus.textContent = '정상';
            ecStatus.className = 'sensor-status ok';
        }

        // pH 바
        const phTargetBar = document.getElementById('ph-target-bar');
        const phCurrentBar = document.getElementById('ph-current-bar');
        const phTargetText = document.getElementById('ph-target');
        const phStatus = document.getElementById('ph-status');

        phTargetText.textContent = `목표: ${t.ph_min} - ${t.ph_max}`;

        // 바 위치 계산 (4-9 pH 범위 기준)
        const phMin = 4, phMax = 9;
        const phTargetLeft = ((t.ph_min - phMin) / (phMax - phMin)) * 100;
        const phTargetWidth = ((t.ph_max - t.ph_min) / (phMax - phMin)) * 100;
        const phCurrentPos = ((sensor.ph - phMin) / (phMax - phMin)) * 100;

        phTargetBar.style.left = `${phTargetLeft}%`;
        phTargetBar.style.width = `${phTargetWidth}%`;
        phCurrentBar.style.left = `${Math.min(Math.max(phCurrentPos, 0), 100)}%`;

        // 상태 표시
        if (sensor.ph < t.ph_min) {
            phStatus.textContent = '낮음';
            phStatus.className = 'sensor-status low';
        } else if (sensor.ph > t.ph_max) {
            phStatus.textContent = '높음';
            phStatus.className = 'sensor-status high';
        } else {
            phStatus.textContent = '정상';
            phStatus.className = 'sensor-status ok';
        }
    }

    updateControllerStatus(controller) {
        if (!controller) return;

        this.autoControl = controller.is_enabled;
        document.getElementById('auto-toggle').checked = this.autoControl;
        this.updateAutoStatus();

        if (controller.targets) {
            this.targets = controller.targets;
        }

        // 펌프 상태는 pump_action 이벤트로만 업데이트 (수동 제어 상태 유지)
        // 자동 제어 시에만 컨트롤러 상태 반영
        if (this.autoControl) {
            const pumpStates = controller.pump_states || {};
            this.updatePumpButton('pump-ab-btn', pumpStates.nutrient_ab);
            this.updatePumpButton('pump-c-btn', pumpStates.nutrient_c);
            this.updatePumpButton('pump-mixer-btn', pumpStates.mixer);
        }

        // 쿨다운
        const cooldowns = controller.pump_cooldowns || {};
        this.updateCooldown('ab-cooldown', cooldowns.nutrient_ab);
        this.updateCooldown('c-cooldown', cooldowns.nutrient_c);
    }

    updatePumpButton(id, isActive) {
        const btn = document.getElementById(id);
        if (btn) {
            btn.classList.toggle('active', isActive);
            const statusEl = btn.querySelector('.pump-status');
            if (statusEl) {
                statusEl.textContent = isActive ? 'ON' : 'OFF';
            }
            // 펌프 상태 변수도 업데이트
            const pump = btn.dataset.pump;
            if (pump && this.pumpStates.hasOwnProperty(pump)) {
                this.pumpStates[pump] = isActive;
            }
        }
    }

    updateCooldown(id, seconds) {
        const el = document.getElementById(id);
        if (el) {
            if (seconds > 0) {
                el.textContent = `대기: ${Math.ceil(seconds)}초`;
            } else {
                el.textContent = '';
            }
        }
    }

    updateDetection(data) {
        this.currentStage = data.stage;
        this.targets = data.targets;

        const stageNames = {
            'germination': '발아기',
            'growth': '생장기',
            'harvest': '수확기',
            'unknown': '대기'
        };

        // 카메라 오버레이 배지
        const badge = document.getElementById('detection-badge');
        badge.className = `badge ${data.stage}`;
        badge.querySelector('.stage').textContent = stageNames[data.stage] || '대기';
        badge.querySelector('.conf').textContent = data.confidence ?
            `${(data.confidence * 100).toFixed(0)}%` : '-';

        // 상세 정보
        document.getElementById('det-stage').textContent = stageNames[data.stage] || '-';
        document.getElementById('det-conf').textContent = data.confidence ?
            `${(data.confidence * 100).toFixed(0)}%` : '-';

        // 바운딩 박스
        if (data.bbox) {
            const [x1, y1, x2, y2] = data.bbox;
            document.getElementById('det-bbox').textContent = `${x2-x1}x${y2-y1}`;
        }

        // 인식 횟수
        this.detectionCount = data.detection_count || this.detectionCount + 1;
        document.getElementById('det-count').textContent = this.detectionCount;

        // 생장단계 카드 활성화
        document.querySelectorAll('.stage-card').forEach(card => {
            card.classList.toggle('active', card.dataset.stage === data.stage);
        });

        this.addLog(`[인식] ${stageNames[data.stage]} (${(data.confidence * 100).toFixed(0)}%)`, 'action');
    }

    updateDetectionStatus(detection) {
        if (detection.current_stage) {
            this.currentStage = detection.current_stage;

            const stageNames = {
                'germination': '발아기',
                'growth': '생장기',
                'harvest': '수확기',
                'unknown': '대기'
            };

            const badge = document.getElementById('detection-badge');
            badge.className = `badge ${detection.current_stage}`;
            badge.querySelector('.stage').textContent = stageNames[detection.current_stage] || '대기';
            badge.querySelector('.conf').textContent = detection.last_confidence ?
                `${(detection.last_confidence * 100).toFixed(0)}%` : '-';

            // 카드 활성화
            document.querySelectorAll('.stage-card').forEach(card => {
                card.classList.toggle('active', card.dataset.stage === detection.current_stage);
            });
        }
    }

    bindEvents() {
        // 자동 제어 토글
        document.getElementById('auto-toggle').addEventListener('change', (e) => {
            this.setAutoControl(e.target.checked);
        });

        // 펌프 버튼 - 변수로 상태 추적
        document.querySelectorAll('.pump-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const pump = btn.dataset.pump;
                const currentState = this.pumpStates[pump] || false;
                const newState = !currentState;
                console.log(`[펌프] ${pump}: ${currentState} -> ${newState}`);
                this.controlPump(pump, newState);
            });
        });
    }

    setAutoControl(enabled) {
        if (!this.isConnected) return;

        this.ws.send(JSON.stringify({
            type: 'set_auto',
            enabled: enabled
        }));

        this.autoControl = enabled;
        this.updateAutoStatus();
        this.addLog(`자동 제어: ${enabled ? '활성화' : '비활성화'}`, 'action');
    }

    controlPump(pump, state) {
        if (!this.isConnected) return;

        // 상태 변수 먼저 업데이트 (다음 클릭 전에 반영)
        this.pumpStates[pump] = state;

        // 즉각적인 UI 업데이트
        const btnIds = { 'nutrient_ab': 'pump-ab-btn', 'nutrient_c': 'pump-c-btn', 'mixer': 'pump-mixer-btn' };
        this.updatePumpButton(btnIds[pump], state);

        this.ws.send(JSON.stringify({
            type: 'manual_pump',
            pump: pump,
            state: state
        }));

        const pumpNames = {
            'nutrient_ab': '양액AB',
            'nutrient_c': '양액C',
            'mixer': '교반기'
        };
        this.addLog(`[수동] ${pumpNames[pump]} ${state ? 'ON' : 'OFF'}`, 'action');
    }

    addLog(message, type = '') {
        const container = document.getElementById('log-container');
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;

        const time = new Date().toLocaleTimeString('ko-KR', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
        entry.textContent = `${time} ${message}`;

        container.insertBefore(entry, container.firstChild);

        // 최대 30개 유지
        while (container.children.length > 30) {
            container.removeChild(container.lastChild);
        }
    }
}

// 앱 시작
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GinsengApp();
});
