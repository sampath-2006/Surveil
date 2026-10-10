import json
import paho.mqtt.client as mqtt
from src.utils.logger import logger

class MQTTAlertBackend:
    """
    Lightweight IoT-native alert backend using MQTT.
    Ideal for Raspberry Pi communicating with smart home hubs (Home Assistant)
    or remote monitoring servers.
    """
    def __init__(self, broker_address="localhost", port=1883, topic="vlm/danger_alerts"):
        self.broker_address = broker_address
        self.port = port
        self.topic = topic
        self.client = mqtt.Client(client_id="rpi5_vlm_system")
        
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start()
            logger.info(f"MQTT Backend connected to {self.broker_address}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")

    def send_alert(self, severity: str, message: dict):
        """Publishes the danger alert as a JSON payload."""
        payload = {
            "severity": severity,
            "details": message,
            "source": "RPi5_VLM_Node"
        }
        
        try:
            # QoS 1 guarantees at least one delivery
            self.client.publish(self.topic, json.dumps(payload), qos=1)
            logger.info(f"MQTT Alert published to {self.topic}: {severity}")
        except Exception as e:
            logger.error(f"Failed to publish MQTT alert: {e}")
            
    def cleanup(self):
        self.client.loop_stop()
        self.client.disconnect()
