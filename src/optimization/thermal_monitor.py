import time
import threading
from src.utils.logger import logger

class ThermalMonitor:
    """
    Monitors Raspberry Pi CPU temperature to prevent thermal throttling.
    """
    def __init__(self, warning_temp: float = 75.0, throttle_temp: float = 80.0, critical_temp: float = 85.0):
        self.warning_temp = warning_temp
        self.throttle_temp = throttle_temp
        self.critical_temp = critical_temp
        self.is_throttling = False
        
    def get_cpu_temp(self) -> float:
        """Reads CPU temperature from Linux sysfs."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = float(f.read()) / 1000.0
                return temp_c
        except FileNotFoundError:
            # Fallback for non-RPi testing environments (e.g., Windows/Mac)
            return 45.0
        except Exception as e:
            logger.warning(f"Could not read CPU temperature: {e}")
            return 0.0

    def check_thermal_state(self):
        """Checks the thermal state and sets throttling flags."""
        temp = self.get_cpu_temp()
        
        if temp >= self.critical_temp:
            logger.critical(f"CPU Temperature CRITICAL: {temp:.1f}°C. Pausing all inference!")
            self.is_throttling = True
            time.sleep(5) # Forced pause
        elif temp >= self.throttle_temp:
            if not self.is_throttling:
                logger.warning(f"CPU Temperature HIGH: {temp:.1f}°C. Engaging thermal throttle.")
                self.is_throttling = True
        elif temp >= self.warning_temp:
            logger.warning(f"CPU Temperature RISING: {temp:.1f}°C.")
            self.is_throttling = False
        else:
            if self.is_throttling:
                logger.info(f"CPU Temperature normalized: {temp:.1f}°C. Releasing thermal throttle.")
            self.is_throttling = False
            
        return temp
