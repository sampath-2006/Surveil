import os
import time
import cv2
import logging
import numpy as np
from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
from src.optimization.vlm_manager import vlm_manager
from src.mapping.reconstructor import map_reconstructor
from src.utils.logger import logger

app = Flask(__name__)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

API_KEY = os.getenv("SURVEIL_API_KEY", "dev_secret_key")

shared_state_ref = None
danger_event_ref = None
force_mapping_flag = [False]

def require_api_key(func):
    def wrapper(*args, **kwargs):
        provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if provided_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

def init_api(state_ref, event_ref, mapping_flag_ref):
    global shared_state_ref, danger_event_ref, force_mapping_flag
    shared_state_ref = state_ref
    danger_event_ref = event_ref
    force_mapping_flag = mapping_flag_ref

@app.route('/api/state', methods=['GET'])
@require_api_key
def get_state():
    if shared_state_ref is None:
        return jsonify({"error": "State not initialized"}), 500
        
    map_info = map_reconstructor.get_map_info()
    return jsonify({
        "danger_active": shared_state_ref.get("danger_active", False),
        "mapping_active": shared_state_ref.get("mapping_active", False),
        "vlm_loaded": vlm_manager._current_model_name or "None",
        "surveillance_active": True,
        "latest_map": map_info
    })

@app.route('/api/feed', methods=['GET'])
def video_feed():
    """
    Streams the live surveillance camera feed via MJPEG.
    Allows embedding directly via <img src="/api/feed" />.
    """
    # Accept token in header or query parameter
    token = request.headers.get("X-API-Key") or request.args.get("api_key")
    if token and token != API_KEY:
        return Response("Unauthorized", status=401)

    def generate_frames():
        while True:
            frame = None
            if shared_state_ref is not None:
                frame = shared_state_ref.get("current_frame")
                
            if frame is None:
                # Fallback frame while waiting for initial camera frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "CONNECTING TO CAMERA FEED...", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033) # ~30 FPS

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_frame', methods=['GET'])
def current_frame():
    """Returns a single latest snapshot JPEG from the surveillance stream."""
    token = request.headers.get("X-API-Key") or request.args.get("api_key")
    if token and token != API_KEY:
        return Response("Unauthorized", status=401)
        
    frame = None
    if shared_state_ref is not None:
        frame = shared_state_ref.get("current_frame")
        
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "NO SIGNAL", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return Response(status=500)

@app.route('/api/map_info', methods=['GET'])
@require_api_key
def get_map_info():
    """Returns 3D map location, Stanford PLY format specs, point count, and extents."""
    info = map_reconstructor.get_map_info()
    return jsonify(info)

@app.route('/api/map_snapshot', methods=['GET'])
def get_map_snapshot():
    """Serves rasterized 2D novel view snapshot PNGs generated from the 3D map."""
    view = request.args.get("view", "latest")
    version = request.args.get("version")
    
    if view == "latest" or not view:
        filepath = os.path.join(map_reconstructor.snapshots_dir, "latest_render.png")
    elif version:
        filepath = os.path.join(map_reconstructor.snapshots_dir, f"map_v{version}_{view}.png")
    else:
        filepath = os.path.join(map_reconstructor.snapshots_dir, f"map_v{map_reconstructor.version}_{view}.png")
        
    if not os.path.exists(filepath):
        # Fallback to any available snapshot
        latest_fallback = os.path.join(map_reconstructor.snapshots_dir, "latest_render.png")
        if os.path.exists(latest_fallback):
            filepath = latest_fallback
        else:
            # Generate on the fly
            map_reconstructor.render_novel_view(view_name="isometric_projection")
            filepath = latest_fallback
    return send_file(os.path.abspath(filepath), mimetype='image/png')

@app.route('/api/render_map_view', methods=['POST'])
@require_api_key
def render_map_view():
    """
    On-demand Novel View Synthesis:
    Rasterizes and renders a 2D perspective snapshot from the active 3D map.
    """
    data = request.get_json(silent=True) or {}
    view_name = data.get("view_name", "isometric_projection")
    
    logger.info(f"Triggered on-demand 3D Map rendering for view '{view_name}'")
    map_reconstructor.render_novel_view(view_name=view_name)
    
    info = map_reconstructor.get_map_info()
    return jsonify({
        "status": "success",
        "view_rendered": view_name,
        "image_url": f"/api/map_snapshot?view={view_name}&t={int(time.time()*1000)}",
        "map_info": info
    })

@app.route('/api/trigger_danger', methods=['POST'])
@require_api_key
def trigger_danger():
    if not shared_state_ref["danger_active"]:
        shared_state_ref["danger_signal"] = True
        shared_state_ref["danger_active"] = True
        danger_event_ref.set()
        return jsonify({"status": "danger_triggered"})
    return jsonify({"status": "already_active"}), 400

@app.route('/api/trigger_mapping', methods=['POST'])
@require_api_key
def trigger_mapping():
    if not shared_state_ref["danger_active"] and not shared_state_ref.get("mapping_active", False):
        force_mapping_flag[0] = True
        return jsonify({"status": "mapping_triggered"})
    return jsonify({"status": "busy"}), 400

def start_api_server(state_ref, event_ref, mapping_flag_ref):
    init_api(state_ref, event_ref, mapping_flag_ref)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
