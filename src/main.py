import os
import sys
# Add project root to python path so 'import src' works correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import threading
from src.utils.logger import logger
from src.core.config import load_config
from src.capture.frame_engine import FrameEngine
from src.optimization.thermal_monitor import ThermalMonitor
from src.detection.trigger import DangerTrigger
from src.detection.vlm_danger import VLMDangerAnalyzer
from src.mapping.vlm_object_detector import VLMObjectDetector
from src.mapping.vlm_depth_estimator import VLMDepthEstimator
from src.mapping.reconstructor import MapReconstructor

# Global shared state
shared_state = {
    "current_frame": None,
    "keyframe_buffer": [], # Stores recent keyframes for mapping
    "danger_signal": False,
    "danger_active": False
}
state_lock = threading.Lock()
danger_event = threading.Event()

def surveillance_loop(config, thermal_monitor):
    """Thread 1: Continuous surveillance loop (Camera -> Trigger)"""
    logger.info("Starting surveillance loop")
    engine = FrameEngine(
        camera_id=0,
        resolution=tuple(config.capture.resolution),
        fps=config.capture.fps,
        ssim_threshold=config.surveillance.ssim_threshold
    )
    
    trigger = DangerTrigger(
        model_path=config.surveillance.trigger_model,
        confidence_threshold=config.surveillance.trigger_confidence
    )
    trigger.load()
    
    try:
        for frame, is_keyframe in engine.get_frames():
            with state_lock:
                shared_state["current_frame"] = frame.copy()
                
            if is_keyframe:
                with state_lock:
                    # Keep last N keyframes for mapping
                    shared_state["keyframe_buffer"].append(frame.copy())
                    if len(shared_state["keyframe_buffer"]) > config.mapping.keyframes_per_cycle:
                        shared_state["keyframe_buffer"].pop(0)

                thermal_monitor.check_thermal_state()
                
                if trigger.infer(frame) and not shared_state["danger_active"]:
                    logger.warning("Danger trigger fired! Signaling VLM thread.")
                    with state_lock:
                        shared_state["danger_signal"] = True
                        shared_state["danger_active"] = True
                    danger_event.set()
    except Exception as e:
        logger.error("Error in surveillance loop", exc_info=True)
    finally:
        engine.stop()

def danger_analysis_loop(config, thermal_monitor, alert_manager):
    """Thread 2: Danger analysis via Moondream 2B (Async)"""
    logger.info("Starting danger analysis listener")
    
    prompt = """Analyze this frame. Classify: SAFE, SMOKE, FIRE, CONFLICT, OTHER_DANGER.
    Respond in JSON: {"type": "...", "location": "...", "severity": "...", "action": "..."}"""
    
    vlm_analyzer = VLMDangerAnalyzer(model_path=config.danger.model, prompt=prompt)
    
    while True:
        danger_event.wait()
        danger_event.clear()
        
        if thermal_monitor.is_throttling:
            logger.warning("Thermal throttle active, delaying VLM danger analysis...")
            time.sleep(2)
            
        with state_lock:
            frame_to_analyze = shared_state["current_frame"]
            
        if frame_to_analyze is not None:
            logger.info("Analyzing frame for danger...")
            result = vlm_analyzer.analyze(frame_to_analyze)
            
            if result and result.get("severity") in ["HIGH", "CRITICAL"]:
                logger.critical(f"🚨 CONFIRMED DANGER: {result}")
                alert_manager.dispatch_alert(result)
            else:
                logger.info(f"Danger dismissed or low severity: {result}")
            
        with state_lock:
            shared_state["danger_signal"] = False
            shared_state["danger_active"] = False

def mapping_loop(config, thermal_monitor):
    """Thread 3: 3D Mapping via Florence-2 and DepthVLM (Scheduled)"""
    interval = config.mapping.interval_seconds
    logger.info(f"Starting scheduled 3D mapping loop (Interval: {interval}s)")
    
    object_detector = VLMObjectDetector(model_path=config.mapping.object_detection.model)
    depth_estimator = VLMDepthEstimator(model_path=config.mapping.depth_estimation.model)
    reconstructor = MapReconstructor()
    
    while True:
        time.sleep(interval)
        
        with state_lock:
            if shared_state["danger_active"]:
                logger.info("Danger analysis active, postponing mapping cycle.")
                continue
            # Grab keyframes for processing
            keyframes = list(shared_state["keyframe_buffer"])
                
        if thermal_monitor.is_throttling:
            logger.warning("Thermal throttle active, skipping mapping cycle to allow cooling.")
            continue
            
        if not keyframes:
            logger.info("No keyframes available for mapping. Skipping cycle.")
            continue
                
        logger.info(f"Starting 3D mapping cycle with {len(keyframes)} keyframes...")
        
        # Step 1: Detect objects (Loads Florence-2, then unloads)
        detections = object_detector.detect(keyframes)
        
        # Step 2: Estimate metric depth (Loads DepthVLM, then unloads)
        depth_maps = depth_estimator.estimate_depth(keyframes)
        
        # Step 3: Reconstruct 3D Map
        reconstructor.rebuild_map(keyframes, detections, depth_maps)

def main():
    logger.info("Initializing VLM-Based Danger Detection & 3D Mapping System (RPi 5)")
    config = load_config()
    thermal_monitor = ThermalMonitor(
        warning_temp=config.thermal.warning_temp_c,
        throttle_temp=config.thermal.throttle_temp_c,
        critical_temp=config.thermal.critical_temp_c
    )
    
    from src.alerting.alert_manager import AlertManager
    alert_manager = AlertManager()
    
    t_surveillance = threading.Thread(target=surveillance_loop, args=(config, thermal_monitor), daemon=True, name="Surveillance")
    t_danger = threading.Thread(target=danger_analysis_loop, args=(config, thermal_monitor, alert_manager), daemon=True, name="DangerAnalysis")
    t_mapping = threading.Thread(target=mapping_loop, args=(config, thermal_monitor), daemon=True, name="Mapping")
    
    t_surveillance.start()
    t_danger.start()
    t_mapping.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down system")
        alert_manager.shutdown()

if __name__ == "__main__":
    main()
