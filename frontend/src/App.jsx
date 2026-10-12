import React, { useState, useEffect, useRef } from 'react';
import './index.css';

const API_URL = "http://127.0.0.1:5000/api";

function App() {
  const [logs, setLogs] = useState([]);
  const [systemState, setSystemState] = useState({
    surveillanceActive: false,
    dangerTriggered: false,
    mappingActive: false,
    vlmLoaded: "None",
    connected: false
  });
  
  const [mapInfo, setMapInfo] = useState({
    version: 1,
    format: "Stanford Polygon (.ply) 3D Point Cloud",
    ply_path: "data/maps/map_v1.ply",
    file_size_kb: 128.0,
    vertices_count: 32400,
    bounds: { x_range_m: [-3.5, 3.5], y_range_m: [-1.8, 1.8], z_range_m: [0.0, 3.0] },
    detected_objects: [
      { label: "desk", depth_m: 2.1 },
      { label: "doorway", depth_m: 3.8 }
    ],
    last_rebuilt: "Awaiting cycle..."
  });

  const [selectedView, setSelectedView] = useState("isometric_projection");
  const [renderTimestamp, setRenderTimestamp] = useState(Date.now());
  const [isRendering, setIsRendering] = useState(false);
  const [feedError, setFeedError] = useState(false);

  const logContainerRef = useRef(null);
  const prevStateRef = useRef(systemState);

  const addLog = (msg, type = "INFO") => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { time, msg, type }]);
  };

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // Polling backend state & map info
  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch(`${API_URL}/state`, {
          headers: { 'X-API-Key': 'dev_secret_key' }
        });
        if (!res.ok) throw new Error("API response not ok");
        const data = await res.json();
        
        setSystemState(prev => {
          const newState = {
            surveillanceActive: data.surveillance_active,
            dangerTriggered: data.danger_active,
            mappingActive: data.mapping_active,
            vlmLoaded: data.vlm_loaded,
            connected: true
          };
          
          const previousState = prevStateRef.current;
          if (!previousState.dangerTriggered && newState.dangerTriggered) {
             addLog("MobileNetV3 Triggered! Suspected danger detected. Invoking Moondream 2B.", "WARN");
          }
          if (previousState.dangerTriggered && !newState.dangerTriggered) {
             addLog("Danger Analysis resolved.", "SUCCESS");
          }
          if (!previousState.mappingActive && newState.mappingActive) {
             addLog("Scheduled 3D Mapping Cycle Started (Florence-2 + Depth).", "INFO");
          }
          if (previousState.mappingActive && !newState.mappingActive) {
             addLog("3D Mapping Cycle complete. Stanford .ply point cloud updated.", "SUCCESS");
             setRenderTimestamp(Date.now());
          }
          if (previousState.vlmLoaded !== newState.vlmLoaded && newState.vlmLoaded !== "None") {
             addLog(`VLMManager memory swap: ${newState.vlmLoaded} mapped into RAM.`, "INFO");
          }
          
          prevStateRef.current = newState;
          return newState;
        });

        if (data.latest_map) {
          setMapInfo(data.latest_map);
        }
        setFeedError(false);
      } catch (err) {
        if (systemState.connected) {
          addLog("Lost connection to Python Backend.", "DANGER");
          setSystemState(prev => ({...prev, connected: false}));
        }
      }
    };

    const interval = setInterval(fetchState, 1000);
    return () => clearInterval(interval);
  }, [systemState.connected]);

  const simulateDanger = async () => {
    try {
      await fetch(`${API_URL}/trigger_danger`, { 
          method: 'POST',
          headers: { 'X-API-Key': 'dev_secret_key' }
      });
    } catch (e) {
      console.error(e);
    }
  };

  const simulateMapping = async () => {
    try {
      await fetch(`${API_URL}/trigger_mapping`, { 
          method: 'POST',
          headers: { 'X-API-Key': 'dev_secret_key' }
      });
    } catch (e) {
      console.error(e);
    }
  };

  // Render on-demand 2D view from 3D map
  const handleRenderView = async (viewName = selectedView) => {
    setIsRendering(true);
    addLog(`Initiating Novel View Synthesis: Rasterizing view '${viewName}' from 3D map...`, "INFO");
    try {
      const res = await fetch(`${API_URL}/render_map_view`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-API-Key': 'dev_secret_key' 
        },
        body: JSON.stringify({ view_name: viewName })
      });
      const data = await res.json();
      if (data.status === "success") {
        setRenderTimestamp(Date.now());
        if (data.map_info) setMapInfo(data.map_info);
        addLog(`Successfully synthesized 2D frame from 3D Map v${data.map_info?.version || mapInfo.version} (${viewName})!`, "SUCCESS");
      }
    } catch (e) {
      addLog(`Failed to render view from 3D map: ${e.message}`, "DANGER");
    } finally {
      setIsRendering(false);
    }
  };

  const switchView = (viewName) => {
    setSelectedView(viewName);
    handleRenderView(viewName);
  };

  return (
    <>
      <header>
        <h1>Surveil Dashboard</h1>
        <p style={{color: 'var(--text-secondary)'}}>
          Edge-Optimized Multi-Modal VLM Surveillance & 3D Spatial Mapping • 
          <span style={{color: systemState.connected ? 'var(--success-color)' : 'var(--danger-color)', marginLeft: '10px'}}>
            {systemState.connected ? '● Backend Connected' : '○ Backend Disconnected'}
          </span>
        </p>
      </header>

      <div className="dashboard-grid">
        <main className="main-column">
          
          <div className="feeds-row">
            {/* PANEL 1: REAL-TIME SURVEILLANCE FEED */}
            <div className="glass-panel">
              <div className="feed-header-row">
                <h2>📷 Real-Time Surveillance</h2>
                <div className={`feed-badge ${systemState.dangerTriggered ? 'danger' : ''}`}>
                  {systemState.dangerTriggered ? '🚨 DANGER DETECTED' : '● LIVE FEED'}
                </div>
              </div>

              <div 
                className="feed-container" 
                style={{ 
                  borderColor: systemState.dangerTriggered ? 'var(--danger-color)' : 'rgba(255,255,255,0.1)',
                  boxShadow: systemState.dangerTriggered ? '0 0 25px rgba(255,42,95,0.4)' : 'none'
                }}
              >
                {systemState.surveillanceActive && <div className="scanning-line"></div>}
                
                {systemState.connected && !feedError ? (
                  <img 
                    src={`${API_URL}/feed?api_key=dev_secret_key`} 
                    alt="Real-time surveillance camera stream"
                    className="live-media-frame"
                    onError={() => setFeedError(true)}
                  />
                ) : (
                  <div className="feed-placeholder">
                    {systemState.connected ? (
                      <>
                        <span style={{color: 'var(--accent-color)'}}>Loading MJPEG Camera Stream...</span>
                        <span>30 FPS • MobileNetV3 INT8 Gated</span>
                      </>
                    ) : (
                      <span>Awaiting Backend Connection...</span>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* PANEL 2: 3D SCENE MAP & NOVEL VIEW RENDERER */}
            <div className="glass-panel">
              <div className="feed-header-row">
                <h2>🗺️ 3D Scene Map</h2>
                <div className="feed-badge">
                  {systemState.mappingActive ? 'REBUILDING MAP...' : 'SYNTHESIZED 2D VIEW'}
                </div>
              </div>

              {/* View Angle Selector Tabs */}
              <div className="view-selector">
                <button 
                  className={`view-btn ${selectedView === 'isometric_projection' ? 'active' : ''}`}
                  onClick={() => switchView('isometric_projection')}
                  disabled={isRendering}
                >
                  Isometric 3D
                </button>
                <button 
                  className={`view-btn ${selectedView === 'perspective_center' ? 'active' : ''}`}
                  onClick={() => switchView('perspective_center')}
                  disabled={isRendering}
                >
                  Perspective Center
                </button>
                <button 
                  className={`view-btn ${selectedView === 'top_down_ortho' ? 'active' : ''}`}
                  onClick={() => switchView('top_down_ortho')}
                  disabled={isRendering}
                >
                  Top-Down Ortho
                </button>
              </div>

              <div className="feed-container">
                {systemState.connected ? (
                  <img 
                    src={`${API_URL}/map_snapshot?view=${selectedView}&api_key=dev_secret_key&t=${renderTimestamp}`} 
                    alt="Synthesized 2D perspective rendered from 3D map"
                    className="live-media-frame"
                  />
                ) : (
                  <div className="feed-placeholder">
                    <span>Awaiting 3D Reconstruction...</span>
                  </div>
                )}
              </div>

              {/* 3D Map Storage & Format Inspector */}
              <div className="map-metadata-box">
                <div className="meta-row">
                  <span className="meta-label">Storage Location</span>
                  <span className="meta-value accent">{mapInfo.ply_path || 'data/maps/map_v1.ply'}</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">File Format</span>
                  <span className="meta-value highlight">{mapInfo.format || 'Stanford PLY (.ply) Point Cloud'}</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">Spatial Vertices</span>
                  <span className="meta-value">{(mapInfo.vertices_count || 32400).toLocaleString()} points ({mapInfo.file_size_kb || 128} KB)</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">Bounding Extents</span>
                  <span className="meta-value">X: [-3.5m, +3.5m] | Y: [-1.8m, +1.8m] | Z: [0.0m, +3.0m]</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">Detected Objects</span>
                  <span className="meta-value">
                    {mapInfo.detected_objects ? mapInfo.detected_objects.map(o => `${o.label} (${o.depth_m || 2}m)`).join(', ') : 'desk (2.1m), doorway (3.8m)'}
                  </span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">Last Reconstructed</span>
                  <span className="meta-value">{mapInfo.last_rebuilt || 'Initial Baseline'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* SYSTEM LOGS */}
          <div className="glass-panel">
            <h2>System Logs</h2>
            <div className="log-container" ref={logContainerRef}>
              {logs.length === 0 && (
                <div className="log-entry" style={{color: 'var(--text-secondary)'}}>
                  System initialized. Monitoring camera feed and 3D map cycles...
                </div>
              )}
              {logs.map((log, i) => (
                <div key={i} className="log-entry">
                  <span className="log-time">[{log.time}]</span>
                  <span style={{
                    color: log.type === 'DANGER' ? 'var(--danger-color)' : 
                           log.type === 'WARN' ? 'var(--warning-color)' : 
                           log.type === 'SUCCESS' ? 'var(--success-color)' : 
                           'var(--text-primary)'
                  }}>
                    {log.msg}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </main>

        <aside className="side-column">
          <div className="glass-panel">
            <h2>System State</h2>
            <ul className="status-list">
              <li className="status-item">
                <span>Surveillance Loop</span>
                <div className={`status-dot ${systemState.surveillanceActive ? 'active' : ''}`}></div>
              </li>
              <li className="status-item">
                <span>Danger Analysis (Async)</span>
                <div className={`status-dot ${systemState.dangerTriggered ? 'danger' : ''}`}></div>
              </li>
              <li className="status-item">
                <span>3D Mapping (Schedule)</span>
                <div className={`status-dot ${systemState.mappingActive ? 'active' : ''}`}></div>
              </li>
            </ul>

            <div style={{marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: '1px solid var(--surface-border)'}}>
              <h3>Memory State (16GB)</h3>
              <div style={{marginTop: '0.5rem', color: 'var(--accent-color)', fontFamily: 'var(--font-mono)'}}>
                VLM Loaded: {systemState.vlmLoaded}
              </div>
            </div>
          </div>

          <div className="glass-panel">
            <h2>Backend Controls</h2>
            <p style={{fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem'}}>
              Inject triggers into the live Python backend via the API.
            </p>
            <div className="controls" style={{flexDirection: 'column'}}>
              {/* ON-DEMAND NOVEL VIEW RENDER BUTTON */}
              <button 
                className="accent"
                onClick={() => handleRenderView(selectedView)}
                disabled={!systemState.connected || isRendering}
                style={{opacity: (!systemState.connected || isRendering) ? 0.6 : 1}}
              >
                {isRendering ? 'Rendering 3D Novel View...' : '📷 Render Snapshot from 3D Map'}
              </button>

              <button 
                className="danger" 
                onClick={simulateDanger}
                disabled={!systemState.connected || systemState.dangerTriggered}
                style={{opacity: (!systemState.connected || systemState.dangerTriggered) ? 0.5 : 1}}
              >
                Trigger MobileNet Anomaly
              </button>
              
              <button 
                onClick={simulateMapping}
                disabled={!systemState.connected || systemState.mappingActive || systemState.dangerTriggered}
                style={{opacity: (!systemState.connected || systemState.mappingActive || systemState.dangerTriggered) ? 0.5 : 1}}
              >
                Force 3D Map Cycle
              </button>
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}

export default App;
