"""
YOLO 새싹인삼 생장단계 인식 모듈
YOLO Ginseng Growth Stage Detector
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from PIL import Image, ImageDraw, ImageFont

class GrowthStage(Enum):
    UNKNOWN = "unknown"
    GERMINATION = "germination"  # 발아기
    GROWTH = "growth"            # 생장기
    HARVEST = "harvest"          # 수확기


@dataclass
class Detection:
    stage: GrowthStage
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float


class GinsengDetector:
    def __init__(self, model_path: str = "models/ginseng_growth.pt",
                 confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = "cpu"  # 기본값, load_model에서 업데이트
        self.is_loaded = False
        self.last_detection: Optional[Detection] = None
        self.last_detections: List[Detection] = []  # 모든 인식 결과
        self.detection_history: List[Detection] = []
        # 모델 클래스 이름 (한글) -> GrowthStage 매핑
        self.class_mapping = {
            "발아기": GrowthStage.GERMINATION,
            "생장기": GrowthStage.GROWTH,
            "수확기": GrowthStage.HARVEST,
        }
        # 한글 폰트 로드
        self.font = None
        self._load_font()

    def _load_font(self):
        """한글 폰트 로드"""
        font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
        ]
        for path in font_paths:
            if Path(path).exists():
                try:
                    self.font = ImageFont.truetype(path, 20)
                    print(f"[YOLO] 폰트 로드: {path}")
                    return
                except:
                    pass
        print("[YOLO] 한글 폰트 없음 - 영문 사용")
        self.font = ImageFont.load_default()

    def _detect_device(self) -> str:
        """최적의 디바이스 감지 (M4 Mac = MPS)"""
        import torch
        if torch.backends.mps.is_available():
            print("[YOLO] Apple Silicon MPS 가속 사용")
            return "mps"
        elif torch.cuda.is_available():
            print("[YOLO] NVIDIA CUDA 가속 사용")
            return "cuda"
        else:
            print("[YOLO] CPU 사용")
            return "cpu"

    def load_model(self) -> bool:
        """YOLO 모델 로드 (M4 Mac 최적화)"""
        try:
            from ultralytics import YOLO

            # 최적 디바이스 감지
            self.device = self._detect_device()

            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"[YOLO] 모델 로드됨: {self.model_path}")
            else:
                print(f"[YOLO] 모델 파일 없음. 기본 YOLOv8n 사용")
                self.model = YOLO("yolov8n.pt")
                print("[YOLO] 새싹인삼 데이터셋으로 학습이 필요합니다")

            # 워밍업 추론 (첫 추론 속도 개선)
            import numpy as np
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy, device=self.device, verbose=False)
            print(f"[YOLO] 워밍업 완료 (device={self.device})")

            self.is_loaded = True
            return True
        except ImportError:
            print("[YOLO] ultralytics 패키지가 필요합니다: pip install ultralytics")
            return False
        except Exception as e:
            print(f"[YOLO] 모델 로드 실패: {e}")
            return False

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """프레임에서 모든 생장단계 인식 (MPS 가속 + 최적화)"""
        if not self.is_loaded or self.model is None:
            return None

        try:
            # M4 Mac MPS 가속 추론 (최적화)
            results = self.model(
                frame,
                device=self.device,
                imgsz=320,  # 320px로 빠른 추론 (640 대비 4배 빠름)
                verbose=False
            )

            detections = []
            best_detection = None
            best_confidence = 0

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    if conf < self.confidence_threshold:
                        continue

                    cls_id = int(box.cls[0])
                    class_name = self.model.names.get(cls_id, "")
                    stage = self.class_mapping.get(class_name)
                    if stage is None:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detection = Detection(
                        stage=stage,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        timestamp=time.time()
                    )
                    detections.append(detection)

                    if conf > best_confidence:
                        best_confidence = conf
                        best_detection = detection

            # 모든 인식 결과 저장
            self.last_detections = detections

            if best_detection:
                self.last_detection = best_detection
                self.detection_history.append(best_detection)
                if len(self.detection_history) > 100:
                    self.detection_history = self.detection_history[-100:]

            return best_detection

        except Exception as e:
            print(f"[YOLO] 인식 오류: {e}")
            return None

    def get_stable_stage(self, window: int = 5) -> GrowthStage:
        """최근 N개 인식 결과 중 가장 많은 단계 반환 (안정화)"""
        if len(self.detection_history) < window:
            if self.last_detection:
                return self.last_detection.stage
            return GrowthStage.UNKNOWN

        recent = self.detection_history[-window:]
        stage_counts = {}
        for det in recent:
            stage = det.stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        return max(stage_counts, key=stage_counts.get)

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """프레임에 모든 인식 결과 표시 (한글 지원)"""
        if not self.last_detections:
            return frame

        # OpenCV BGR -> PIL RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 단계별 색상 (RGB)
        colors = {
            GrowthStage.GERMINATION: (0, 255, 0),    # 초록
            GrowthStage.GROWTH: (255, 255, 0),       # 노랑
            GrowthStage.HARVEST: (255, 165, 0),      # 주황
            GrowthStage.UNKNOWN: (128, 128, 128),    # 회색
        }

        stage_names = {
            GrowthStage.GERMINATION: "발아기",
            GrowthStage.GROWTH: "생장기",
            GrowthStage.HARVEST: "수확기",
            GrowthStage.UNKNOWN: "미인식",
        }

        for det in self.last_detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.stage, (255, 255, 255))
            label = f"{stage_names[det.stage]} {det.confidence:.0%}"

            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 라벨 배경
            text_bbox = draw.textbbox((x1, y1 - 25), label, font=self.font)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2,
                           text_bbox[2] + 2, text_bbox[3] + 2], fill=color)

            # 라벨 텍스트
            draw.text((x1, y1 - 25), label, font=self.font, fill=(0, 0, 0))

        # PIL RGB -> OpenCV BGR
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def draw_detection(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """단일 인식 결과 표시 (호환성 유지) - 모든 결과 표시로 변경"""
        return self.draw_detections(frame)

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        stage = self.get_stable_stage()
        return {
            "is_loaded": self.is_loaded,
            "current_stage": stage.value,
            "current_stage_ko": {
                GrowthStage.GERMINATION: "발아기",
                GrowthStage.GROWTH: "생장기",
                GrowthStage.HARVEST: "수확기",
                GrowthStage.UNKNOWN: "미인식",
            }.get(stage, "미인식"),
            "last_confidence": self.last_detection.confidence if self.last_detection else 0,
            "detection_count": len(self.detection_history),
            "current_objects": len(self.last_detections),  # 현재 프레임 인식 객체 수
        }
