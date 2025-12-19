"""
YOLO 모델 학습 관리자
Training Manager for YOLO model
"""

import os
import shutil
import asyncio
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import cv2
import numpy as np

# GitHub 기본 모델 URL (terrawave/ginseng-sprout-detection)
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/terrawave/ginseng-sprout-detection/master/best.pt"


@dataclass
class LabeledImage:
    image_id: str
    filename: str
    labels: List[Dict]  # [{"class_id": 0, "bbox": [x_center, y_center, w, h]}]
    created_at: str
    labeled: bool = False


class TrainingManager:
    def __init__(self, base_path: str = "data/training"):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.labels_path = self.base_path / "labels"
        self.models_path = Path("models")

        # 클래스 정의
        self.classes = ["발아기", "생장기", "수확기"]
        self.class_map = {name: idx for idx, name in enumerate(self.classes)}

        # 학습 상태
        self.is_training = False
        self.training_progress = 0
        self.training_status = "idle"
        self.current_epoch = 0
        self.total_epochs = 50

        # 디렉토리 생성
        self._init_directories()

        # 이미지 메타데이터
        self.metadata_file = self.base_path / "metadata.json"
        self.images_meta: Dict[str, LabeledImage] = {}
        self._load_metadata()

    def _init_directories(self):
        """디렉토리 초기화"""
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # train/val 분리용
        (self.base_path / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.base_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.base_path / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.base_path / "val" / "labels").mkdir(parents=True, exist_ok=True)

    def _load_metadata(self):
        """메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for img_id, meta in data.items():
                        self.images_meta[img_id] = LabeledImage(**meta)
            except Exception as e:
                print(f"[Training] 메타데이터 로드 실패: {e}")

    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            data = {k: asdict(v) for k, v in self.images_meta.items()}
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Training] 메타데이터 저장 실패: {e}")

    def download_base_model(self) -> Optional[str]:
        """GitHub에서 기본 모델 다운로드"""
        try:
            base_model_path = self.models_path / "ginseng_base.pt"
            print(f"[Training] GitHub에서 기본 모델 다운로드 중...")
            print(f"[Training] URL: {GITHUB_MODEL_URL}")

            # 다운로드
            urllib.request.urlretrieve(GITHUB_MODEL_URL, str(base_model_path))

            # 파일 확인
            if base_model_path.exists() and base_model_path.stat().st_size > 1000000:  # 1MB 이상
                print(f"[Training] 기본 모델 다운로드 완료: {base_model_path}")
                return str(base_model_path)
            else:
                print(f"[Training] 다운로드 실패 - 파일 크기 이상")
                return None

        except Exception as e:
            print(f"[Training] GitHub 모델 다운로드 실패: {e}")
            return None

    def capture_image(self, frame: np.ndarray) -> Optional[str]:
        """프레임 캡처 및 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_id = f"img_{timestamp}"
            filename = f"{image_id}.jpg"
            filepath = self.images_path / filename

            cv2.imwrite(str(filepath), frame)

            # 메타데이터 추가
            self.images_meta[image_id] = LabeledImage(
                image_id=image_id,
                filename=filename,
                labels=[],
                created_at=datetime.now().isoformat(),
                labeled=False
            )
            self._save_metadata()

            print(f"[Training] 이미지 캡처: {filename}")
            return image_id
        except Exception as e:
            print(f"[Training] 캡처 실패: {e}")
            return None

    def save_label(self, image_id: str, labels: List[Dict]) -> bool:
        """라벨 저장 (YOLO 포맷)"""
        try:
            if image_id not in self.images_meta:
                return False

            meta = self.images_meta[image_id]
            meta.labels = labels
            meta.labeled = len(labels) > 0

            # YOLO 포맷으로 라벨 파일 저장
            label_filename = meta.filename.replace('.jpg', '.txt')
            label_filepath = self.labels_path / label_filename

            with open(label_filepath, 'w') as f:
                for label in labels:
                    class_id = label.get('class_id', 0)
                    bbox = label.get('bbox', [0.5, 0.5, 0.1, 0.1])
                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            self._save_metadata()
            print(f"[Training] 라벨 저장: {image_id} ({len(labels)}개 객체)")
            return True
        except Exception as e:
            print(f"[Training] 라벨 저장 실패: {e}")
            return False

    def get_images_list(self) -> List[Dict]:
        """이미지 목록 반환"""
        return [
            {
                "image_id": meta.image_id,
                "filename": meta.filename,
                "labeled": meta.labeled,
                "label_count": len(meta.labels),
                "created_at": meta.created_at
            }
            for meta in sorted(
                self.images_meta.values(),
                key=lambda x: x.created_at,
                reverse=True
            )
        ]

    def get_image(self, image_id: str) -> Optional[Dict]:
        """이미지 상세 정보"""
        if image_id not in self.images_meta:
            return None
        meta = self.images_meta[image_id]
        return {
            "image_id": meta.image_id,
            "filename": meta.filename,
            "labels": meta.labels,
            "labeled": meta.labeled,
            "created_at": meta.created_at,
            "image_path": f"/api/training/image/{image_id}"
        }

    def delete_image(self, image_id: str) -> bool:
        """이미지 삭제"""
        try:
            if image_id not in self.images_meta:
                return False

            meta = self.images_meta[image_id]

            # 파일 삭제
            img_file = self.images_path / meta.filename
            label_file = self.labels_path / meta.filename.replace('.jpg', '.txt')

            if img_file.exists():
                img_file.unlink()
            if label_file.exists():
                label_file.unlink()

            del self.images_meta[image_id]
            self._save_metadata()

            print(f"[Training] 이미지 삭제: {image_id}")
            return True
        except Exception as e:
            print(f"[Training] 삭제 실패: {e}")
            return False

    def _prepare_dataset(self, val_ratio: float = 0.2):
        """학습용 데이터셋 준비 (train/val 분리)"""
        labeled_images = [m for m in self.images_meta.values() if m.labeled]

        if len(labeled_images) < 5:
            raise ValueError(f"라벨링된 이미지가 부족합니다 (최소 5개, 현재 {len(labeled_images)}개)")

        # 셔플
        import random
        random.shuffle(labeled_images)

        # train/val 분리
        val_count = max(1, int(len(labeled_images) * val_ratio))
        val_images = labeled_images[:val_count]
        train_images = labeled_images[val_count:]

        # 파일 복사
        for split, images in [("train", train_images), ("val", val_images)]:
            split_img_path = self.base_path / split / "images"
            split_lbl_path = self.base_path / split / "labels"

            # 기존 파일 삭제
            for f in split_img_path.glob("*"):
                f.unlink()
            for f in split_lbl_path.glob("*"):
                f.unlink()

            # 파일 복사
            for meta in images:
                src_img = self.images_path / meta.filename
                src_lbl = self.labels_path / meta.filename.replace('.jpg', '.txt')

                if src_img.exists():
                    shutil.copy(src_img, split_img_path / meta.filename)
                if src_lbl.exists():
                    shutil.copy(src_lbl, split_lbl_path / meta.filename.replace('.jpg', '.txt'))

        print(f"[Training] 데이터셋 준비 완료: train={len(train_images)}, val={len(val_images)}")
        return len(train_images), len(val_images)

    def _create_data_yaml(self) -> str:
        """data.yaml 생성"""
        yaml_path = self.base_path / "data.yaml"

        content = f"""# Ginseng Growth Dataset
path: {self.base_path.absolute()}
train: train/images
val: val/images

nc: {len(self.classes)}
names: {self.classes}
"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(yaml_path)

    async def start_training(self, epochs: int = 50, base_model: str = "yolov8n.pt", use_pretrained: bool = True) -> bool:
        """학습 시작"""
        if self.is_training:
            print("[Training] 이미 학습 중입니다")
            return False

        try:
            self.is_training = True
            self.training_status = "preparing"
            self.training_progress = 0
            self.total_epochs = epochs
            self.current_epoch = 0

            # 데이터셋 준비
            train_count, val_count = self._prepare_dataset()

            # data.yaml 생성
            data_yaml = self._create_data_yaml()

            self.training_status = "training"

            # 비동기로 학습 실행
            asyncio.create_task(self._run_training(data_yaml, epochs, base_model))

            return True
        except Exception as e:
            self.is_training = False
            self.training_status = f"error: {e}"
            print(f"[Training] 학습 시작 실패: {e}")
            return False

    async def _run_training(self, data_yaml: str, epochs: int, base_model: str):
        """실제 학습 실행 (비동기)"""
        try:
            from ultralytics import YOLO
            from ultralytics.utils import callbacks

            model = None
            model_source = "none"

            # 1. GitHub에서 기본 모델 다운로드 시도
            github_model_path = self.download_base_model()
            if github_model_path:
                try:
                    temp_model = YOLO(github_model_path)
                    num_classes = len(temp_model.names)
                    if num_classes == 3:  # 발아기, 생장기, 수확기
                        model = temp_model
                        model_source = "github"
                        print(f"[Training] GitHub 모델로 파인튜닝: {github_model_path} (클래스 {num_classes}개)")
                    else:
                        print(f"[Training] GitHub 모델 클래스 수 불일치 ({num_classes}개)")
                except Exception as e:
                    print(f"[Training] GitHub 모델 로드 실패: {e}")

            # 2. GitHub 실패 시 로컬 모델 사용
            if model is None:
                existing_model = Path("models/ginseng_growth.pt")
                if existing_model.exists():
                    try:
                        temp_model = YOLO(str(existing_model))
                        num_classes = len(temp_model.names)
                        if num_classes == 3:
                            model = temp_model
                            model_source = "local"
                            print(f"[Training] 로컬 모델로 파인튜닝: {existing_model} (클래스 {num_classes}개)")
                    except Exception as e:
                        print(f"[Training] 로컬 모델 로드 실패: {e}")

            # 3. 모두 실패 시 YOLOv8n 사용
            if model is None:
                print(f"[Training] YOLOv8 pretrained 모델 로드: {base_model}")
                model = YOLO(base_model)
                model_source = "pretrained"

            print(f"[Training] 학습 시작: epochs={epochs}, source={model_source}")

            # 진행 상황 추적을 위한 참조
            manager = self

            # 콜백 함수 정의
            def on_train_epoch_end(trainer):
                manager.current_epoch = trainer.epoch + 1
                manager.training_progress = int((trainer.epoch + 1) / epochs * 100)
                manager.training_status = f"학습 중: {manager.current_epoch}/{epochs} 에포크"
                metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
                box_loss = metrics.get('train/box_loss', 0)
                cls_loss = metrics.get('train/cls_loss', 0)
                print(f"[Training] Epoch {manager.current_epoch}/{epochs} - box_loss: {box_loss:.4f}, cls_loss: {cls_loss:.4f}")

            def on_train_batch_end(trainer):
                # 배치 단위로 더 세밀한 진행률 계산
                if hasattr(trainer, 'epoch') and hasattr(trainer, 'batch_i'):
                    batch_progress = (trainer.batch_i + 1) / len(trainer.train_loader) if hasattr(trainer, 'train_loader') else 0
                    epoch_progress = (trainer.epoch + batch_progress) / epochs * 100
                    manager.training_progress = min(int(epoch_progress), 99)

            # 학습 실행 (별도 스레드에서)
            loop = asyncio.get_event_loop()

            def train_sync():
                # 콜백 등록
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                model.add_callback("on_train_batch_end", on_train_batch_end)

                results = model.train(
                    data=data_yaml,
                    epochs=epochs,
                    imgsz=640,
                    batch=8,
                    device="mps",  # M4 Mac
                    project=str(self.models_path),
                    name="ginseng_train",
                    exist_ok=True,
                    verbose=True
                )
                return results

            # 백그라운드에서 실행
            results = await loop.run_in_executor(None, train_sync)

            # 최신 모델 복사
            best_model = self.models_path / "ginseng_train" / "weights" / "best.pt"
            if best_model.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_model_name = f"ginseng_growth_{timestamp}.pt"
                shutil.copy(best_model, self.models_path / new_model_name)

                # 기본 모델로 설정
                shutil.copy(best_model, self.models_path / "ginseng_growth.pt")
                print(f"[Training] 새 모델 저장: {new_model_name}")

            self.training_status = "completed"
            self.training_progress = 100
            print("[Training] 학습 완료!")

        except Exception as e:
            self.training_status = f"error: {e}"
            print(f"[Training] 학습 실패: {e}")
        finally:
            self.is_training = False

    def get_training_status(self) -> Dict:
        """학습 상태 반환"""
        return {
            "is_training": self.is_training,
            "status": self.training_status,
            "progress": self.training_progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs
        }

    def get_models_list(self) -> List[Dict]:
        """모델 목록"""
        models = []
        for f in self.models_path.glob("*.pt"):
            stat = f.stat()
            models.append({
                "name": f.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_active": f.name == "ginseng_growth.pt"
            })
        return sorted(models, key=lambda x: x["modified"], reverse=True)

    def switch_model(self, model_name: str) -> bool:
        """모델 교체"""
        try:
            model_path = self.models_path / model_name
            if not model_path.exists():
                return False

            # 현재 모델 백업
            current = self.models_path / "ginseng_growth.pt"
            if current.exists():
                backup_name = f"ginseng_growth_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                shutil.copy(current, self.models_path / backup_name)

            # 새 모델로 교체
            shutil.copy(model_path, current)
            print(f"[Training] 모델 교체: {model_name}")
            return True
        except Exception as e:
            print(f"[Training] 모델 교체 실패: {e}")
            return False

    def get_stats(self) -> Dict:
        """통계 정보"""
        total = len(self.images_meta)
        labeled = sum(1 for m in self.images_meta.values() if m.labeled)
        return {
            "total_images": total,
            "labeled_images": labeled,
            "unlabeled_images": total - labeled,
            "classes": self.classes,
            "models_count": len(list(self.models_path.glob("*.pt")))
        }
