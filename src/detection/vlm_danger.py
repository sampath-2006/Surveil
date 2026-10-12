import json
import time
import cv2
import base64
from src.utils.logger import logger
from src.optimization.vlm_manager import vlm_manager

DANGER_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["SAFE", "SMOKE", "FIRE", "CONFLICT", "INTRUSION", "OTHER_DANGER"]},
        "location": {"type": "string"},
        "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
        "action": {"type": "string"}
    },
    "required": ["type", "location", "severity", "action"]
}

class VLMDangerAnalyzer:
    """
    Wrapper for Moondream 2B using llama.cpp.
    Executes heavy contextual analysis when triggered, with GBNF constrained decoding.
    """
    def __init__(self, model_path: str, prompt: str):
        self.model_path = model_path
        self.prompt = prompt
        self._grammar = None
        self._init_grammar()
        
    def _init_grammar(self):
        try:
            from llama_cpp import LlamaGrammar
            self._grammar = LlamaGrammar.from_json_schema(json.dumps(DANGER_SCHEMA))
            logger.info("Initialized GBNF grammar constraint for Moondream 2B JSON decoding.")
        except Exception as e:
            logger.warning(f"Could not initialize LlamaGrammar ({e}). Falling back to json_object format.")
            self._grammar = None

    def _frame_to_base64(self, frame) -> str:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
        
    def analyze(self, frame) -> dict:
        """
        Loads Moondream, analyzes the frame using constrained GBNF decoding,
        returns guaranteed valid JSON result, and records latency.
        """
        logger.info("Requesting Moondream 2B from VLM Manager...")
        engine = vlm_manager.load_moondream(self.model_path)
        
        logger.info("Executing Moondream 2B inference with GBNF grammar constraint...")
        start_time = time.time()
        
        if engine == "mock_moondream":
            time.sleep(2) # Realistic prototype latency
            result = {
                "type": "FIRE",
                "location": "bottom-right",
                "severity": "HIGH",
                "action": "Trigger alarm and notify admin"
            }
        else:
            try:
                # Real llama.cpp inference with GBNF constrained grammar
                b64_img = self._frame_to_base64(frame)
                kwargs = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]
                        }
                    ],
                    "max_tokens": 128
                }
                if self._grammar is not None:
                    kwargs["grammar"] = self._grammar
                else:
                    kwargs["response_format"] = {"type": "json_object"}
                    
                response = engine.create_chat_completion(**kwargs)
                content = response['choices'][0]['message']['content']
                result = json.loads(content)
            except Exception as e:
                logger.error(f"Llama.cpp inference error: {e}")
                result = {
                    "type": "OTHER_DANGER",
                    "location": "unknown",
                    "severity": "LOW",
                    "action": "Inspection required",
                    "error": str(e)
                }
        
        # Enforce schema integrity
        for key, default_val in [("type", "SAFE"), ("location", "center"), ("severity", "LOW"), ("action", "none")]:
            if key not in result:
                result[key] = default_val

        inference_time = time.time() - start_time
        logger.info(f"VLM Inference completed in {inference_time:.2f}s | Result: {result.get('type')} ({result.get('severity')})")
        result["latency_s"] = round(inference_time, 2)
        
        return result

