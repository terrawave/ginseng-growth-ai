"""
USB 카메라 캡처 모듈
Camera Capture Module
"""

import cv2
import asyncio
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import threading

class CameraManager:
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.last_frame = None
        self.last_capture_time = 0
        self.lock = threading.Lock()

    def start(self) -> bool:
        """카메라 시작"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"[Camera] 카메라 {self.device_id} 열기 실패")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.is_running = True
            print(f"[Camera] 카메라 시작됨 ({self.width}x{self.height})")
            return True
        except Exception as e:
            print(f"[Camera] 시작 오류: {e}")
            return False

    def stop(self):
        """카메라 정지"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[Camera] 카메라 정지됨")

    def get_frame(self):
        """현재 프레임 가져오기"""
        if not self.cap or not self.is_running:
            return None

        with self.lock:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.last_capture_time = time.time()
                return frame
        return None

    def get_frame_base64(self) -> Optional[str]:
        """Base64 인코딩된 프레임 가져오기 (웹 전송용)"""
        frame = self.get_frame()
        if frame is None:
            return None

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')

    def save_capture(self, folder: str = "data/captures") -> Optional[str]:
        """현재 프레임 저장"""
        frame = self.get_frame()
        if frame is None:
            return None

        Path(folder).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{folder}/capture_{timestamp}.jpg"
        cv2.imwrite(filepath, frame)
        print(f"[Camera] 이미지 저장: {filepath}")
        return filepath


class CameraStreamer:
    """MJPEG 스트리밍용"""
    def __init__(self, camera: CameraManager):
        self.camera = camera

    async def generate_frames(self):
        """프레임 스트리밍 제너레이터"""
        while self.camera.is_running:
            frame = self.camera.get_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.033)  # ~30fps
