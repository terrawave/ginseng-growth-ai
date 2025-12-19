"""
양액 자동 제어 모듈
Nutrient Auto Control Module

EC/pH 자동 조절:
- EC 낮음 → 양액AB 작동 → EC 상승
- pH 높음 → 양액C 작동 → pH 하강
"""

import asyncio
import time
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
from modules.yolo_detector import GrowthStage

# 생장단계별 목표값
STAGE_TARGETS = {
    GrowthStage.GERMINATION: {  # 발아기
        "ec_min": 0.8, "ec_max": 1.2,
        "ph_min": 5.8, "ph_max": 6.2,
    },
    GrowthStage.GROWTH: {  # 생장기
        "ec_min": 1.4, "ec_max": 1.8,
        "ph_min": 5.5, "ph_max": 6.0,
    },
    GrowthStage.HARVEST: {  # 수확기
        "ec_min": 1.2, "ec_max": 1.5,
        "ph_min": 5.5, "ph_max": 6.0,
    },
    GrowthStage.UNKNOWN: {  # 미인식 시 안전값
        "ec_min": 1.0, "ec_max": 1.4,
        "ph_min": 5.5, "ph_max": 6.0,
    },
}


class PumpType(Enum):
    NUTRIENT_AB = "nutrient_ab"  # 양액AB - EC 상승
    NUTRIENT_C = "nutrient_c"    # 양액C - pH 하강
    MIXER = "mixer"              # 교반기


@dataclass
class ControlAction:
    pump: PumpType
    reason: str
    current_value: float
    target_min: float
    target_max: float
    timestamp: float


class NutrientController:
    def __init__(self,
                 pump_duration: float = 5.0,
                 pump_cooldown: float = 60.0,
                 ec_tolerance: float = 0.1,
                 ph_tolerance: float = 0.1):
        self.pump_duration = pump_duration  # 펌프 작동 시간 (초)
        self.pump_cooldown = pump_cooldown  # 펌프 재작동 대기 시간 (초)
        self.ec_tolerance = ec_tolerance
        self.ph_tolerance = ph_tolerance

        self.current_stage = GrowthStage.UNKNOWN
        self.is_enabled = False  # 자동 제어 활성화 여부
        self.is_running = False

        # 펌프 상태
        self.pump_states = {
            PumpType.NUTRIENT_AB: False,
            PumpType.NUTRIENT_C: False,
            PumpType.MIXER: False,
        }
        self.last_pump_times = {
            PumpType.NUTRIENT_AB: 0,
            PumpType.NUTRIENT_C: 0,
            PumpType.MIXER: 0,
        }

        # 콜백 함수 (실제 펌프 제어)
        self.pump_callback: Optional[Callable] = None

        # 로그
        self.action_history: list = []

    def set_pump_callback(self, callback: Callable):
        """펌프 제어 콜백 설정"""
        self.pump_callback = callback

    def set_stage(self, stage: GrowthStage):
        """생장단계 설정"""
        if stage != self.current_stage:
            print(f"[Controller] 생장단계 변경: {self.current_stage.value} → {stage.value}")
            self.current_stage = stage

    def get_targets(self) -> Dict:
        """현재 단계의 목표값 반환"""
        return STAGE_TARGETS.get(self.current_stage, STAGE_TARGETS[GrowthStage.UNKNOWN])

    def enable(self):
        """자동 제어 활성화"""
        self.is_enabled = True
        print("[Controller] 자동 제어 활성화")

    def disable(self):
        """자동 제어 비활성화"""
        self.is_enabled = False
        print("[Controller] 자동 제어 비활성화")

    def can_activate_pump(self, pump: PumpType) -> bool:
        """펌프 활성화 가능 여부 (쿨다운 체크)"""
        last_time = self.last_pump_times.get(pump, 0)
        return time.time() - last_time >= self.pump_cooldown

    async def activate_pump(self, pump: PumpType, reason: str):
        """펌프 작동 (양액펌프는 교반기도 함께 작동)"""
        if self.pump_states[pump]:
            return  # 이미 작동 중

        if not self.can_activate_pump(pump):
            remaining = self.pump_cooldown - (time.time() - self.last_pump_times[pump])
            print(f"[Controller] {pump.value} 쿨다운 중 ({remaining:.0f}초 남음)")
            return

        # 양액 펌프인 경우 교반기도 함께 작동
        with_mixer = pump in [PumpType.NUTRIENT_AB, PumpType.NUTRIENT_C]

        print(f"[Controller] {pump.value} 작동 시작 - {reason}")
        self.pump_states[pump] = True
        self.last_pump_times[pump] = time.time()

        if with_mixer:
            self.pump_states[PumpType.MIXER] = True
            print(f"[Controller] 교반기 함께 작동")

        # 실제 펌프 제어
        if self.pump_callback:
            try:
                if pump == PumpType.NUTRIENT_AB:
                    await self.pump_callback("slave1", 1, True)  # slave1 릴레이1
                elif pump == PumpType.NUTRIENT_C:
                    await self.pump_callback("slave1", 2, True)  # slave1 릴레이2

                # 양액 펌프와 함께 교반기 켜기
                if with_mixer:
                    await self.pump_callback("slave1", 3, True)  # slave1 릴레이3 = 교반기
            except Exception as e:
                print(f"[Controller] 펌프 제어 오류: {e}")

        # 지정 시간 후 정지
        await asyncio.sleep(self.pump_duration)

        self.pump_states[pump] = False
        print(f"[Controller] {pump.value} 작동 정지")

        if with_mixer:
            self.pump_states[PumpType.MIXER] = False
            print(f"[Controller] 교반기 정지")

        if self.pump_callback:
            try:
                if pump == PumpType.NUTRIENT_AB:
                    await self.pump_callback("slave1", 1, False)
                elif pump == PumpType.NUTRIENT_C:
                    await self.pump_callback("slave1", 2, False)

                # 양액 펌프와 함께 교반기 끄기
                if with_mixer:
                    await self.pump_callback("slave1", 3, False)
            except Exception as e:
                print(f"[Controller] 펌프 정지 오류: {e}")

    def check_and_control(self, ec: float, ph: float) -> Optional[ControlAction]:
        """
        EC/pH 값 확인 및 제어 필요 여부 판단

        순서: EC 먼저 맞추고 → pH 조절
        - EC가 목표 범위 밖이면 EC만 제어 (pH 무시)
        - EC가 목표 범위 안이면 pH 제어

        Returns:
            ControlAction if action needed, None otherwise
        """
        if not self.is_enabled:
            return None

        targets = self.get_targets()
        action = None

        # EC가 목표 범위 안인지 확인
        ec_in_range = targets["ec_min"] <= ec <= targets["ec_max"]

        # 1단계: EC 체크 - EC가 낮으면 먼저 EC부터 맞춤
        if ec < targets["ec_min"] - self.ec_tolerance:
            if self.can_activate_pump(PumpType.NUTRIENT_AB):
                action = ControlAction(
                    pump=PumpType.NUTRIENT_AB,
                    reason=f"EC 낮음 ({ec:.2f} < {targets['ec_min']:.2f})",
                    current_value=ec,
                    target_min=targets["ec_min"],
                    target_max=targets["ec_max"],
                    timestamp=time.time()
                )
                self.action_history.append(action)
                print(f"[Controller] EC 조절 중 - pH 조절 대기")
            return action  # EC 조절 중에는 pH 건드리지 않음

        # 2단계: EC가 범위 내일 때만 pH 체크
        if ec_in_range:
            if ph > targets["ph_max"] + self.ph_tolerance:
                if self.can_activate_pump(PumpType.NUTRIENT_C):
                    action = ControlAction(
                        pump=PumpType.NUTRIENT_C,
                        reason=f"pH 높음 ({ph:.2f} > {targets['ph_max']:.2f})",
                        current_value=ph,
                        target_min=targets["ph_min"],
                        target_max=targets["ph_max"],
                        timestamp=time.time()
                    )
                    self.action_history.append(action)
                    print(f"[Controller] EC 정상 → pH 조절 시작")
        else:
            if ph > targets["ph_max"] + self.ph_tolerance:
                print(f"[Controller] pH 높음 대기 중 (EC 먼저 조절 필요: {ec:.2f})")

        return action

    async def control_loop(self, get_sensor_data: Callable):
        """자동 제어 루프"""
        self.is_running = True
        print("[Controller] 자동 제어 루프 시작")

        while self.is_running:
            try:
                if self.is_enabled:
                    # 센서 데이터 가져오기
                    data = await get_sensor_data()
                    if data:
                        ec = data.get("ec", 0)
                        ph = data.get("ph", 7)

                        # 제어 필요 여부 확인
                        action = self.check_and_control(ec, ph)
                        if action:
                            await self.activate_pump(action.pump, action.reason)

                await asyncio.sleep(10)  # 10초마다 체크

            except Exception as e:
                print(f"[Controller] 루프 오류: {e}")
                await asyncio.sleep(5)

        print("[Controller] 자동 제어 루프 종료")

    def stop(self):
        """제어 루프 정지"""
        self.is_running = False

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        targets = self.get_targets()
        return {
            "is_enabled": self.is_enabled,
            "current_stage": self.current_stage.value,
            "targets": targets,
            "pump_states": {k.value: v for k, v in self.pump_states.items()},
            "pump_cooldowns": {
                k.value: max(0, self.pump_cooldown - (time.time() - self.last_pump_times[k]))
                for k in PumpType
            },
            "action_count": len(self.action_history),
            "last_action": self.action_history[-1].reason if self.action_history else None,
        }
