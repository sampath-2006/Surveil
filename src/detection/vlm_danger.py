import json
import time
import cv2
import base64
from src.utils.logger import logger
from src.optimization.vlm_manager import vlm_manager

class VLMDangerAnalyzer:
    """
    Wrapper for Moondream 2B using llama.cpp.
    Executes heavy contextual analysis when triggered.
    """
    def __init__(self, model_path: str, prompt: str):
        self.model_path = model_path
        self.prompt = prompt
        
    def _frame_to_base64(self, frame) -> str:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
        
    def analyze(self, frame) -> dict:
        """
        Loads Moondream, analyzes the frame, returns JSON result, and unloads.
        """
        logger.info("Requesting Moondream 2B from VLM Manager...")
        engine = vlm_manager.load_moondream(self.model_path)
        
        logger.info("Executing Moondream 2B inference...")
        start_time = time.time()
        
        if engine == "mock_moondream":
            time.sleep(10) # Simulate RPi inference latency
            result = {
                "type": "FIRE",
                "location": "bottom-right",
                "severity": "HIGH",
                "action": "Trigger alarm and notify admin"
            }
        else:
            try:
                # Real llama.cpp inference
                b64_img = self._frame_to_base64(frame)
                response = engine.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]
                        }
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response['choices'][0]['message']['content']
                result = json.loads(content)
            except Exception as e:
                logger.error(f"Llama.cpp inference error: {e}")
                result = {"severity": "LOW", "error": str(e)}
        
        inference_time = time.time() - start_time
        logger.info(f"VLM Inference completed in {inference_time:.2f}s")
        result["latency_s"] = round(inference_time, 2)
        
        return result
