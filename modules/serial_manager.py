"""
ESP32 데이터 매니저 (terracube-server API 연동)
Data Manager - connects to terracube-server API
"""

import asyncio
import aiohttp
import time
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass


# terracube-server 설정
TERRACUBE_API = "http://localhost:8000"
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwidXNlcm5hbWUiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc2NjY3MDg4NH0.2lhOboALwKGkIfOcz_qIgKeo2DwefkHVjNwfWKREiMY"


@dataclass
class SensorData:
    ec: float = 0.0
    ph: float = 7.0
    water_temp: float = 25.0
    tof_distance: float = 0.0
    flow_rate: float = 0.0
    pressure: float = 0.0
    timestamp: float = 0.0


class SerialManager:
    """terracube-server API를 통해 ESP32 데이터 접근"""

    def __init__(self, api_url: str = TERRACUBE_API, token: str = AUTH_TOKEN):
        self.api_url = api_url
        self.token = token
        self.is_connected = False
        self.is_running = False

        # 센서 데이터
        self.sensor_data = SensorData()
        self.relay_states = {i: False for i in range(1, 17)}
        self.slave1_relay_states = {i: False for i in range(1, 5)}

        # 콜백
        self.on_data_callback: Optional[Callable] = None

        # HTTP 세션
        self.session: Optional[aiohttp.ClientSession] = None

    def _get_headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    async def _fetch_data(self) -> Optional[Dict]:
        """terracube-server에서 실시간 데이터 가져오기"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(
                f"{self.api_url}/api/data/realtime",
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            print(f"[API] 데이터 가져오기 실패: {e}")
        return None

    async def _send_control(self, device: str, action: str, value: Any) -> bool:
        """terracube-server로 제어 명령 전송"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            payload = {
                "device": device,
                "action": action,
                "value": value
            }

            async with self.session.post(
                f"{self.api_url}/api/control",
                headers=self._get_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    print(f"[API] 제어 전송: {device}/{action}={value}")
                    return True
                else:
                    print(f"[API] 제어 실패: {resp.status}")
        except Exception as e:
            print(f"[API] 제어 오류: {e}")
        return False

    async def update_sensor_data(self):
        """센서 데이터 업데이트"""
        data = await self._fetch_data()
        if not data:
            return

        # slave1 센서 데이터 추출
        slave1 = data.get("esp", {}).get("slave1", {})
        sensors = slave1.get("sensors", {})

        if sensors:
            self.sensor_data.ph = sensors.get("ph", 7.0)
            # EC: µS/cm → mS/cm 변환 (1000으로 나눔)
            raw_ec = sensors.get("ec", 0.0)
            self.sensor_data.ec = raw_ec / 1000.0
            self.sensor_data.water_temp = sensors.get("water_temp", 25.0)
            self.sensor_data.pressure = sensors.get("pressure", 0.0)
            self.sensor_data.tof_distance = sensors.get("tof_a", 0.0)
            self.sensor_data.flow_rate = sensors.get("flow_rate", 0.0)
            self.sensor_data.timestamp = time.time()

            if self.on_data_callback:
                self.on_data_callback(self.get_sensor_dict())

        # 릴레이 상태 업데이트
        slave1_relays = slave1.get("relays", {})
        self.slave1_relay_states[1] = slave1_relays.get("nutrient_ab", False)
        self.slave1_relay_states[2] = slave1_relays.get("nutrient_c", False)

        self.is_connected = True

    def set_slave1_relay(self, num: int, state: bool) -> str:
        """슬레이브1 릴레이 제어 (양액 펌프) - 비동기로 전송"""
        action_map = {
            1: "nutrient_ab",
            2: "nutrient_c",
            3: "mixer",
            4: "spare"
        }
        action = action_map.get(num, f"r{num}")

        # 비동기 전송을 위해 태스크 생성
        asyncio.create_task(self._send_control("slave1", action, "on" if state else "off"))
        self.slave1_relay_states[num] = state
        return f"slave1/{action}={'on' if state else 'off'}"

    def set_master_relay(self, channel: int, state: bool) -> str:
        """마스터 릴레이 제어"""
        action_map = {
            1: "cube1_led1", 2: "cube1_led2", 3: "cube1_led3", 4: "cube1_led4",
            5: "cube2_led1", 6: "cube2_led2", 7: "cube2_led3", 8: "cube2_led4",
            9: "tank_valve",
            10: "valve1", 11: "valve2", 12: "valve3", 13: "valve4",
            14: "cube1_fan", 15: "cube2_fan", 16: "spare"
        }
        action = action_map.get(channel, f"m{channel}")

        asyncio.create_task(self._send_control("master", action, "on" if state else "off"))
        self.relay_states[channel] = state
        return f"master/{action}={'on' if state else 'off'}"

    async def data_loop(self):
        """데이터 업데이트 루프"""
        while self.is_running:
            try:
                await self.update_sensor_data()
                await asyncio.sleep(2)  # 2초마다 업데이트
            except Exception as e:
                print(f"[API] 루프 오류: {e}")
                await asyncio.sleep(5)

    def start(self) -> bool:
        """시작 (실제 루프는 main.py에서 asyncio로 실행)"""
        self.is_running = True
        print(f"[API] terracube-server 연동 시작: {self.api_url}")
        return True

    def stop(self):
        """정지"""
        self.is_running = False
        if self.session:
            asyncio.create_task(self.session.close())
        print("[API] 연동 종료")

    def get_sensor_dict(self) -> Dict[str, Any]:
        """센서 데이터 딕셔너리 반환"""
        return {
            "ec": self.sensor_data.ec,
            "ph": self.sensor_data.ph,
            "water_temp": self.sensor_data.water_temp,
            "tof_distance": self.sensor_data.tof_distance,
            "flow_rate": self.sensor_data.flow_rate,
            "pressure": self.sensor_data.pressure,
            "timestamp": self.sensor_data.timestamp,
        }

    def get_status(self) -> Dict:
        """상태 반환"""
        return {
            "is_connected": self.is_connected,
            "api_url": self.api_url,
            "sensor_data": self.get_sensor_dict(),
            "relay_states": self.relay_states,
            "slave1_relay_states": self.slave1_relay_states,
        }
