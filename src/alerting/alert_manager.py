from src.utils.logger import logger
from src.alerting.backends.mqtt_backend import MQTTAlertBackend

class AlertManager:
    """
    Manages distribution of danger alerts to various backends based on severity.
    """
    def __init__(self):
        self.backends = []
        # Register default MQTT backend
        self.register_backend(MQTTAlertBackend())
        
    def register_backend(self, backend):
        self.backends.append(backend)
        logger.info(f"Registered alert backend: {backend.__class__.__name__}")
        
    def dispatch_alert(self, danger_result: dict):
        """
        Dispatches the alert. 
        Only forwards HIGH or CRITICAL severity alerts by default to avoid spam.
        """
        severity = danger_result.get("severity", "LOW")
        
        if severity in ["HIGH", "CRITICAL"]:
            logger.info(f"Dispatching {severity} alert to {len(self.backends)} backends.")
            for backend in self.backends:
                backend.send_alert(severity, danger_result)
        else:
            logger.debug(f"Alert ignored (Severity: {severity} below threshold)")
            
    def shutdown(self):
        for backend in self.backends:
            if hasattr(backend, 'cleanup'):
                backend.cleanup()
