"""
새싹인삼 생육 인식 시스템 - 메인 서버
Ginseng Growth AI - Main Server

USB 카메라 + YOLO 모델로 생장단계 인식 후
EC/pH 자동 제어
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path

from config import CAMERA_CONFIG, CONTROL_CONFIG, SERVER_CONFIG, YOLO_CONFIG
from modules.camera import CameraManager, CameraStreamer
from modules.yolo_detector import GinsengDetector, GrowthStage
from modules.nutrient_controller import NutrientController
from modules.serial_manager import SerialManager
from modules.training_manager import TrainingManager

# 전역 객체
camera = CameraManager(
    device_id=CAMERA_CONFIG["device_id"],
    width=CAMERA_CONFIG["width"],
    height=CAMERA_CONFIG["height"]
)
detector = GinsengDetector(
    model_path=YOLO_CONFIG["model_path"],
    confidence_threshold=YOLO_CONFIG["confidence_threshold"]
)
serial_manager = SerialManager()  # terracube-server API 연동
controller = NutrientController(
    pump_duration=CONTROL_CONFIG["pump_duration"],
    pump_cooldown=CONTROL_CONFIG["pump_cooldown"]
)
training_manager = TrainingManager()  # 학습 관리자

# WebSocket 연결 관리
connected_clients: list[WebSocket] = []


async def broadcast(data: dict):
    """모든 클라이언트에게 브로드캐스트"""
    message = json.dumps(data, ensure_ascii=False)
    for client in connected_clients[:]:
        try:
            await client.send_text(message)
        except:
            connected_clients.remove(client)


async def pump_control_callback(device: str, relay_num: int, state: bool):
    """펌프 제어 콜백 (컨트롤러에서 호출)"""
    if device == "slave1":
        serial_manager.set_slave1_relay(relay_num, state)
        await broadcast({
            "type": "pump_action",
            "device": device,
            "relay": relay_num,
            "state": state
        })


async def detection_loop():
    """카메라 인식 루프 (비동기 최적화)"""
    print("[Main] 인식 루프 시작 (비동기)")
    loop = asyncio.get_event_loop()
    loop_count = 0

    while True:
        try:
            loop_count += 1
            if camera.is_running and detector.is_loaded:
                frame = camera.get_frame()
                if frame is not None:
                    # 블로킹 방지: 별도 스레드에서 YOLO 추론 실행
                    detection = await loop.run_in_executor(None, detector.detect, frame)

                    # 10회마다 상태 출력 (디버그)
                    if loop_count % 10 == 0:
                        print(f"[Debug] 추론 #{loop_count}: detection={detection is not None}")

                    if detection:
                        # 생장단계 업데이트
                        stable_stage = detector.get_stable_stage()
                        controller.set_stage(stable_stage)
                        print(f"[YOLO] 인식: {stable_stage.value} ({detection.confidence:.1%})")

                        # 클라이언트에 전송
                        await broadcast({
                            "type": "detection",
                            "stage": stable_stage.value,
                            "confidence": detection.confidence,
                            "bbox": detection.bbox,
                            "timestamp": detection.timestamp,
                            "detection_count": len(detector.detection_history),
                            "targets": controller.get_targets()
                        })

            await asyncio.sleep(CAMERA_CONFIG.get("capture_interval", 0.5))

        except Exception as e:
            print(f"[Main] 인식 루프 오류: {e}")
            await asyncio.sleep(5)


async def sensor_broadcast_loop():
    """센서 데이터 브로드캐스트 루프"""
    print("[Main] 센서 브로드캐스트 루프 시작")
    while True:
        try:
            if serial_manager.is_connected:
                data = serial_manager.get_sensor_dict()
                status = controller.get_status()

                await broadcast({
                    "type": "sensor_data",
                    "sensor": data,
                    "controller": status,
                    "detection": detector.get_status()
                })

            await asyncio.sleep(2)

        except Exception as e:
            print(f"[Main] 브로드캐스트 오류: {e}")
            await asyncio.sleep(2)


async def get_sensor_data():
    """센서 데이터 반환 (컨트롤러용)"""
    return serial_manager.get_sensor_dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 수명주기 관리"""
    print("=" * 50)
    print("새싹인삼 생육 인식 시스템 시작")
    print("=" * 50)

    # 카메라 시작
    if camera.start():
        print("[Main] 카메라 준비됨")
    else:
        print("[Main] 카메라 시작 실패 - 계속 진행")

    # YOLO 모델 로드
    if detector.load_model():
        print("[Main] YOLO 모델 준비됨")
    else:
        print("[Main] YOLO 모델 로드 실패 - 계속 진행")

    # API 연동 시작
    if serial_manager.start():
        print("[Main] terracube-server API 연동 준비됨")
    else:
        print("[Main] API 연동 실패 - 계속 진행")

    # 컨트롤러 설정
    controller.set_pump_callback(pump_control_callback)

    # 백그라운드 태스크 시작
    detection_task = asyncio.create_task(detection_loop())
    sensor_task = asyncio.create_task(sensor_broadcast_loop())
    control_task = asyncio.create_task(controller.control_loop(get_sensor_data))
    data_task = asyncio.create_task(serial_manager.data_loop())  # API 데이터 루프

    print(f"[Main] 서버 시작: http://localhost:{SERVER_CONFIG['port']}")
    print("=" * 50)

    yield

    # 종료
    print("[Main] 서버 종료 중...")
    detection_task.cancel()
    sensor_task.cancel()
    control_task.cancel()
    data_task.cancel()
    controller.stop()
    camera.stop()
    serial_manager.stop()
    print("[Main] 종료 완료")


app = FastAPI(title="Ginseng Growth AI", lifespan=lifespan)

# 정적 파일
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 페이지"""
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    """전체 상태 조회"""
    return {
        "camera": {"is_running": camera.is_running},
        "detector": detector.get_status(),
        "serial": serial_manager.get_status(),
        "controller": controller.get_status(),
    }


@app.get("/api/camera/frame")
async def get_camera_frame():
    """현재 카메라 프레임 (Base64)"""
    frame_b64 = camera.get_frame_base64()
    if frame_b64:
        return {"frame": frame_b64}
    return JSONResponse({"error": "No frame available"}, status_code=503)


@app.get("/api/camera/stream")
async def camera_stream():
    """MJPEG 스트림"""
    streamer = CameraStreamer(camera)
    return StreamingResponse(
        streamer.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/control/auto")
async def toggle_auto_control(enabled: bool = True):
    """자동 제어 활성화/비활성화"""
    if enabled:
        controller.enable()
    else:
        await controller.disable()
    return {"auto_control": controller.is_enabled}


@app.post("/api/control/pump")
async def manual_pump_control(pump: str, state: bool):
    """수동 펌프 제어"""
    if pump == "nutrient_ab":
        serial_manager.set_slave1_relay(1, state)
    elif pump == "nutrient_c":
        serial_manager.set_slave1_relay(2, state)
    else:
        return JSONResponse({"error": "Unknown pump"}, status_code=400)

    return {"pump": pump, "state": state}


@app.post("/api/stage/set")
async def set_stage_manual(stage: str):
    """수동 생장단계 설정"""
    try:
        growth_stage = GrowthStage(stage)
        controller.set_stage(growth_stage)
        return {"stage": stage, "targets": controller.get_targets()}
    except ValueError:
        return JSONResponse({"error": "Invalid stage"}, status_code=400)


# ============== 학습 API ==============

@app.get("/training", response_class=HTMLResponse)
async def training_page():
    """학습 페이지"""
    return Path("static/training.html").read_text(encoding="utf-8")


@app.post("/api/training/capture")
async def capture_training_image():
    """학습용 이미지 캡처"""
    frame = camera.get_frame()
    if frame is None:
        return JSONResponse({"error": "카메라 프레임 없음"}, status_code=503)

    image_id = training_manager.capture_image(frame)
    if image_id:
        return {"success": True, "image_id": image_id}
    return JSONResponse({"error": "캡처 실패"}, status_code=500)


@app.get("/api/training/images")
async def get_training_images():
    """학습 이미지 목록"""
    return {
        "images": training_manager.get_images_list(),
        "stats": training_manager.get_stats()
    }


@app.get("/api/training/image/{image_id}")
async def get_training_image(image_id: str):
    """학습 이미지 파일 반환"""
    import cv2
    meta = training_manager.images_meta.get(image_id)
    if not meta:
        return JSONResponse({"error": "이미지 없음"}, status_code=404)

    filepath = training_manager.images_path / meta.filename
    if not filepath.exists():
        return JSONResponse({"error": "파일 없음"}, status_code=404)

    img = cv2.imread(str(filepath))
    _, buffer = cv2.imencode('.jpg', img)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@app.get("/api/training/image/{image_id}/info")
async def get_training_image_info(image_id: str):
    """이미지 상세 정보"""
    info = training_manager.get_image(image_id)
    if info:
        return info
    return JSONResponse({"error": "이미지 없음"}, status_code=404)


from fastapi import Request

@app.post("/api/training/label/{image_id}")
async def save_training_label(image_id: str, request: Request):
    """라벨 저장"""
    labels = await request.json()
    if training_manager.save_label(image_id, labels):
        return {"success": True}
    return JSONResponse({"error": "저장 실패"}, status_code=500)


@app.delete("/api/training/image/{image_id}")
async def delete_training_image(image_id: str):
    """이미지 삭제"""
    if training_manager.delete_image(image_id):
        return {"success": True}
    return JSONResponse({"error": "삭제 실패"}, status_code=500)


@app.post("/api/training/start")
async def start_training(epochs: int = 50):
    """학습 시작"""
    if await training_manager.start_training(epochs=epochs):
        return {"success": True, "message": f"{epochs} 에포크 학습 시작"}
    return JSONResponse({"error": "학습 시작 실패"}, status_code=500)


@app.get("/api/training/status")
async def get_training_status():
    """학습 상태"""
    return training_manager.get_training_status()


@app.get("/api/training/models")
async def get_models_list():
    """모델 목록"""
    return {"models": training_manager.get_models_list()}


@app.post("/api/training/switch/{model_name}")
async def switch_model(model_name: str):
    """모델 교체"""
    if training_manager.switch_model(model_name):
        # 디텍터 모델 다시 로드
        detector.model_path = f"models/{model_name}"
        detector.load_model()
        return {"success": True, "message": f"모델 교체: {model_name}"}
    return JSONResponse({"error": "모델 교체 실패"}, status_code=500)


@app.post("/api/training/reload")
async def reload_model():
    """현재 모델 리로드"""
    if detector.load_model():
        return {"success": True, "message": "모델 리로드 완료"}
    return JSONResponse({"error": "리로드 실패"}, status_code=500)


@app.get("/api/system/info")
async def get_system_info():
    """시스템 정보 조회"""
    import platform
    import sys
    import os
    import psutil
    from datetime import datetime

    # CPU 정보
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()

    # 메모리 정보
    memory = psutil.virtual_memory()

    # 디스크 정보
    disk = psutil.disk_usage('/')

    # 프로세스 정보
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info()

    # GPU 정보 (YOLO용)
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    except:
        gpu_available = False
        gpu_name = "N/A"

    return {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": sys.version.split()[0],
        },
        "hardware": {
            "cpu_count": cpu_count,
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_percent": memory.percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_percent": disk.percent,
            "process_memory_mb": round(process_memory.rss / (1024**2), 2),
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
        },
        "ai": {
            "model_loaded": detector.is_loaded,
            "model_path": detector.model_path,
            "confidence_threshold": detector.confidence_threshold,
            "detection_count": len(detector.detection_history),
        },
        "frameworks": {
            "FastAPI": {"version": "0.100+", "description": "비동기 웹 프레임워크"},
            "YOLOv8": {"version": "Ultralytics", "description": "객체 인식 AI 모델"},
            "OpenCV": {"version": "4.x", "description": "컴퓨터 비전 라이브러리"},
            "PyTorch": {"version": "2.x", "description": "딥러닝 프레임워크"},
            "NumPy": {"version": "1.x", "description": "수치 연산 라이브러리"},
            "WebSocket": {"version": "실시간", "description": "양방향 실시간 통신"},
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 연결"""
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] 클라이언트 연결 ({len(connected_clients)}개)")

    try:
        # 초기 상태 전송
        await websocket.send_json({
            "type": "init",
            "detector": detector.get_status(),
            "controller": controller.get_status(),
            "sensor": serial_manager.get_sensor_dict()
        })

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            # 클라이언트 명령 처리
            if msg.get("type") == "set_auto":
                if msg.get("enabled"):
                    controller.enable()
                else:
                    await controller.disable()

            elif msg.get("type") == "manual_pump":
                pump = msg.get("pump")
                state = msg.get("state", False)
                if pump == "nutrient_ab":
                    serial_manager.set_slave1_relay(1, state)
                elif pump == "nutrient_c":
                    serial_manager.set_slave1_relay(2, state)
                elif pump == "mixer":
                    serial_manager.set_slave1_relay(3, state)

                # 모든 클라이언트에게 펌프 상태 브로드캐스트
                await broadcast({
                    "type": "pump_action",
                    "pump": pump,
                    "state": state
                })
                print(f"[WS] 수동 펌프 제어: {pump} = {state}")

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"[WS] 클라이언트 해제 ({len(connected_clients)}개)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_CONFIG["host"], port=SERVER_CONFIG["port"])
