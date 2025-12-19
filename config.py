"""
새싹인삼 생육 인식 시스템 설정
Ginseng Growth AI Configuration
"""

# 생장 단계 정의
GROWTH_STAGES = {
    "germination": {  # 발아기
        "name_ko": "발아기",
        "ec_min": 0.8,
        "ec_max": 1.2,
        "ph_min": 5.8,
        "ph_max": 6.2,
    },
    "growth": {  # 생장기
        "name_ko": "생장기",
        "ec_min": 1.4,
        "ec_max": 1.8,
        "ph_min": 5.5,
        "ph_max": 6.0,
    },
    "harvest": {  # 수확기
        "name_ko": "수확기",
        "ec_min": 1.2,
        "ec_max": 1.5,
        "ph_min": 5.5,
        "ph_max": 6.0,
    },
}

# YOLO 모델 설정
YOLO_CONFIG = {
    "model_path": "models/ginseng_growth.pt",
    "confidence_threshold": 0.3,  # 30% 이상만 인식 (테스트용)
    "class_names": ["germination", "growth", "harvest"],
}

# 카메라 설정
CAMERA_CONFIG = {
    "device_id": 0,  # USB 카메라
    "width": 640,
    "height": 480,
    "fps": 30,
    "capture_interval": 0.5,  # 인식 간격 (초) - 0.5초마다 추론
}

# 제어 설정
CONTROL_CONFIG = {
    "check_interval": 10,  # 센서 체크 간격 (초)
    "pump_duration": 5,  # 펌프 작동 시간 (초)
    "pump_cooldown": 60,  # 펌프 쿨다운 (초)
    "ec_tolerance": 0.1,  # EC 허용 오차
    "ph_tolerance": 0.1,  # pH 허용 오차
}

# 시리얼 통신 설정
SERIAL_CONFIG = {
    "port": "/dev/cu.usbserial-0001",
    "baudrate": 115200,
    "timeout": 1,
}

# 서버 설정
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8001,  # 기존 서버와 다른 포트
}
