import './App.css';
import { useMemo, useState, useRef, useEffect } from 'react';

function App() {
  const [active, setActive] = useState('control');
  const [gestures, setGestures] = useState([]);
  const [newGesture, setNewGesture] = useState('');
  const [mappings, setMappings] = useState({});
  const [actions, setActions] = useState({});
  const [statusItems, setStatusItems] = useState([]);
  const [configError, setConfigError] = useState(null);

  const [liveStream, setLiveStream] = useState(false);
  const [currentPose, setCurrentPose] = useState(null);
  const [currentConfidence, setCurrentConfidence] = useState(null);
  const [annotatedFrame, setAnnotatedFrame] = useState(null);
  const [lastPredictionAt, setLastPredictionAt] = useState(null);
  const [lastAction, setLastAction] = useState(null);
  const [holdDuration, setHoldDuration] = useState(null);
  const [mousePos, setMousePos] = useState(null);

  const [datasetPose, setDatasetPose] = useState('');
  const [datasetLive, setDatasetLive] = useState(false);
  const [datasetRunning, setDatasetRunning] = useState(false);
  const [datasetCount, setDatasetCount] = useState(0);
  const [datasetNewGesture, setDatasetNewGesture] = useState('');

  const [retrainStatus, setRetrainStatus] = useState('Idle');
  const [retrainRunning, setRetrainRunning] = useState(false);

  const controlVideoRef = useRef(null);
  const controlCanvasRef = useRef(null);
  const controlStreamRef = useRef(null);
  const controlIntervalRef = useRef(null);

  const datasetVideoRef = useRef(null);
  const datasetCanvasRef = useRef(null);
  const datasetStreamRef = useRef(null);
  const datasetIntervalRef = useRef(null);

  const apiBase = useMemo(() => {
    if (process.env.REACT_APP_API_BASE) {
      return process.env.REACT_APP_API_BASE;
    }
    if (typeof window !== 'undefined') {
      const origin = window.location.origin;
      if (origin.includes(':3000')) {
        return 'http://localhost:8000';
      }
      return origin;
    }
    return 'http://localhost:8000';
  }, []);

  const appendStatus = (label, kind = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setStatusItems((prev) =>
      [{ id: `${Date.now()}-${Math.random()}`, label, kind, timestamp }, ...prev].slice(
        0,
        6
      )
    );
  };

  const fetchJson = async (path, options = {}) => {
    const response = await fetch(`${apiBase}${path}`, options);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed with ${response.status}`);
    }
    return response.json();
  };

  const loadConfig = async () => {
    try {
      const [gestureData, mappingData, actionData] = await Promise.all([
        fetchJson('/api/gestures'),
        fetchJson('/api/mappings'),
        fetchJson('/api/actions'),
      ]);
      setGestures(gestureData.gestures || []);
      setMappings(mappingData.mappings || {});
      setActions(actionData.actions || {});
      if (!datasetPose && gestureData.gestures?.length) {
        setDatasetPose(gestureData.gestures[0]);
      }
      setConfigError(null);
    } catch (error) {
      appendStatus(error.message || 'Failed to load configuration.', 'error');
      setConfigError(error.message || 'Failed to load configuration.');
    }
  };

  const addGesture = async () => {
    const name = newGesture.trim().toLowerCase();
    if (!name) {
      appendStatus('Gesture name required.', 'error');
      return;
    }
    try {
      const data = await fetchJson('/api/gestures', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const updatedGestures = data.gestures || [];
      setGestures(updatedGestures);
      setNewGesture('');
      appendStatus('Gesture added.', 'success');
      setDatasetPose((current) => current || name);
      await loadConfig();
    } catch (error) {
      appendStatus(error.message || 'Failed to add gesture.', 'error');
    }
  };

  const addDatasetGesture = async () => {
    const name = datasetNewGesture.trim().toLowerCase();
    if (!name) {
      appendStatus('Gesture label required.', 'error');
      return;
    }
    try {
      const data = await fetchJson('/api/gestures', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const updated = data.gestures || [];
      setGestures(updated);
      setDatasetPose(name);
      setDatasetNewGesture('');
      appendStatus('Gesture label added.', 'success');
      await loadConfig();
    } catch (error) {
      appendStatus(error.message || 'Failed to add gesture label.', 'error');
    }
  };

  const deleteGesture = async (gesture) => {
    try {
      const data = await fetchJson(`/api/gestures/${encodeURIComponent(gesture)}`, {
        method: 'DELETE',
      });
      setGestures(data.gestures || []);
      appendStatus('Gesture removed.', 'info');
      await loadConfig();
    } catch (error) {
      appendStatus(error.message || 'Failed to remove gesture.', 'error');
    }
  };

  const updateMapping = async (gesture, action) => {
    try {
      const data = await fetchJson('/api/mappings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gesture, action }),
      });
      setMappings(data.mappings || {});
      appendStatus('Mapping updated.', 'success');
    } catch (error) {
      appendStatus(error.message || 'Failed to update mapping.', 'error');
    }
  };

  const startControlStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      });

      if (controlVideoRef.current) {
        controlVideoRef.current.srcObject = stream;
        controlStreamRef.current = stream;
        setLiveStream(true);
        appendStatus('Control camera started.', 'success');

        controlIntervalRef.current = setInterval(() => {
          captureAndPredict();
        }, 120);
      }
    } catch (error) {
      appendStatus(`Camera error: ${error.message}`, 'error');
    }
  };

  const stopControlStream = () => {
    if (controlStreamRef.current) {
      controlStreamRef.current.getTracks().forEach((track) => track.stop());
      controlStreamRef.current = null;
    }
    if (controlIntervalRef.current) {
      clearInterval(controlIntervalRef.current);
      controlIntervalRef.current = null;
    }
    setLiveStream(false);
    setCurrentPose(null);
    setCurrentConfidence(null);
    setAnnotatedFrame(null);
    setLastAction(null);
    setHoldDuration(null);
    setMousePos(null);
    appendStatus('Control camera stopped.', 'info');
  };

  const captureAndPredict = async () => {
    if (!controlVideoRef.current || !controlCanvasRef.current) return;

    try {
      const canvas = controlCanvasRef.current;
      const context = canvas.getContext('2d');
      const video = controlVideoRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);

      const frameData = canvas.toDataURL('image/jpeg', 0.8);
      const formData = new FormData();
      formData.append('frame_data', frameData);

      const response = await fetch(`${apiBase}/api/predict-frame`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setCurrentPose(result.pose || null);
        setCurrentConfidence(
          typeof result.confidence === 'number' ? result.confidence : null
        );
        if (result.annotated_frame) {
          setAnnotatedFrame(result.annotated_frame);
        }
        
        // Display action (including "no_action")
        if (result.action) {
          setLastAction(result.action);
        }
        
        // Track hold duration for stop_camera
        if (result.hold_duration !== undefined) {
          setHoldDuration(result.hold_duration);
        } else {
          setHoldDuration(null);
        }
        
        // Track mouse position from left hand
        if (result.mouse_pos) {
          setMousePos(result.mouse_pos);
        }
        
        // Stop camera if action was executed
        if (result.stop_camera) {
          stopControlStream();
        }
        
        setLastPredictionAt(Date.now());
      }
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  const startDatasetCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      });

      if (datasetVideoRef.current) {
        datasetVideoRef.current.srcObject = stream;
        datasetStreamRef.current = stream;
        setDatasetLive(true);
        appendStatus('Dataset camera started.', 'success');
      }
    } catch (error) {
      appendStatus(`Dataset camera error: ${error.message}`, 'error');
    }
  };

  const stopDatasetCamera = () => {
    if (datasetStreamRef.current) {
      datasetStreamRef.current.getTracks().forEach((track) => track.stop());
      datasetStreamRef.current = null;
    }
    setDatasetLive(false);
  };

  const startDatasetCapture = () => {
    if (!datasetPose) {
      appendStatus('Select a gesture label before capture.', 'error');
      return;
    }
    if (!datasetLive) {
      appendStatus('Start the dataset camera first.', 'error');
      return;
    }
    if (datasetIntervalRef.current) {
      return;
    }
    setDatasetRunning(true);
    datasetIntervalRef.current = setInterval(() => {
      captureDatasetFrame();
    }, 140);
  };

  const stopDatasetCapture = () => {
    if (datasetIntervalRef.current) {
      clearInterval(datasetIntervalRef.current);
      datasetIntervalRef.current = null;
    }
    setDatasetRunning(false);
  };

  const captureDatasetFrame = async () => {
    if (!datasetVideoRef.current || !datasetCanvasRef.current) return;

    const canvas = datasetCanvasRef.current;
    const context = canvas.getContext('2d');
    const video = datasetVideoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    const formData = new FormData();
    formData.append('gesture', datasetPose);
    formData.append('frame_data', frameData);

    try {
      const response = await fetch(`${apiBase}/api/dataset/capture-frame`, {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const result = await response.json();
        if (result.total_for_pose !== undefined) {
          setDatasetCount(result.total_for_pose);
        }
      }
    } catch (error) {
      console.error('Dataset capture error:', error);
    }
  };

  const clearDatasetSamples = async () => {
    if (!datasetPose) {
      appendStatus('Select a gesture label to clear.', 'error');
      return;
    }
    try {
      const response = await fetch(
        `${apiBase}/api/dataset/${encodeURIComponent(datasetPose)}`,
        { method: 'DELETE' }
      );
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to clear samples.');
      }
      const result = await response.json();
      setDatasetCount(result.total_for_pose || 0);
      appendStatus('Dataset samples cleared.', 'info');
    } catch (error) {
      appendStatus(error.message || 'Failed to clear samples.', 'error');
    }
  };

  const retrainModel = async () => {
    if (retrainRunning) return;
    setRetrainRunning(true);
    setRetrainStatus('Retraining...');
    try {
      const data = await fetchJson('/api/retrain', { method: 'POST' });
      setRetrainStatus(data.message || 'Retraining complete');
      appendStatus('Model retrained.', 'success');
    } catch (error) {
      setRetrainStatus('Retraining failed');
      appendStatus(error.message || 'Retraining failed.', 'error');
    } finally {
      setRetrainRunning(false);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  useEffect(() => {
    return () => {
      if (controlStreamRef.current) {
        controlStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (controlIntervalRef.current) {
        clearInterval(controlIntervalRef.current);
      }
      if (datasetStreamRef.current) {
        datasetStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (datasetIntervalRef.current) {
        clearInterval(datasetIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (active !== 'control' && liveStream) {
      stopControlStream();
    }
    if (active !== 'dataset') {
      stopDatasetCapture();
      if (datasetLive) {
        stopDatasetCamera();
      }
    }
  }, [active]);

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Gesture Control Console</p>
          <h1>Design, train, and deploy custom gestures for desktop control.</h1>
          <p className="lead">
            A UX-focused dashboard to manage your gesture library, map actions, and retrain
            the model in one click.
          </p>
        </div>
        <div className="hero-card">
          <div className="hero-metric">
            <span>API base</span>
            <strong>{apiBase}</strong>
          </div>
          <div className="hero-metric">
            <span>Active view</span>
            <strong>{active}</strong>
          </div>
          <div className="hero-metric">
            <span>Registered gestures</span>
            <strong>{gestures.length}</strong>
          </div>
          {currentPose && (
            <div className="hero-metric">
              <span>Live pose</span>
              <strong>{currentPose}</strong>
            </div>
          )}
        </div>
      </header>

      <section className="workspace">
        <nav className="mode-tabs">
          {[
            { id: 'control', label: 'Live Control' },
            { id: 'gestures', label: 'Gestures' },
            { id: 'dataset', label: 'Make Dataset' },
            { id: 'model', label: 'Model Retrain' },
          ].map((tab) => (
            <button
              key={tab.id}
              className={active === tab.id ? 'tab active' : 'tab'}
              onClick={() => setActive(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="workspace-grid">
          <div className="panel">
            {active === 'control' && (
              <div className="panel-body">
                <h2>Live Control</h2>
                <p>
                  Run gesture recognition live and trigger the mapped desktop action. The
                  system overlays detected landmarks for clarity.
                </p>

                <div className="video-container">
                  <canvas ref={controlCanvasRef} style={{ display: 'none' }} />
                  <video
                    ref={controlVideoRef}
                    autoPlay
                    playsInline
                    className="video-layer"
                  />
                  <img
                    src={annotatedFrame || ''}
                    alt="Annotated prediction"
                    className="video-layer"
                    style={{ display: annotatedFrame ? 'block' : 'none' }}
                  />
                </div>

                {liveStream && (
                  <div className="prediction-display">
                    <div className="prediction-label">
                      <strong>Pose:</strong>{' '}
                      {currentPose || 'No right-hand pose detected yet'}
                    </div>
                    {currentConfidence !== null && (
                      <div className="prediction-confidence">
                        <strong>Confidence:</strong> {(currentConfidence * 100).toFixed(1)}%
                      </div>
                    )}
                    {lastAction && (
                      <div className="prediction-confidence">
                        <strong>Action:</strong>{' '}
                        {lastAction === 'no_action' 
                          ? 'No action' 
                          : (actions[lastAction] || lastAction)}
                      </div>
                    )}
                    {lastAction === 'stop_camera' && holdDuration !== null && (
                      <div className="prediction-confidence hold-progress">
                        <strong>Hold ({holdDuration.toFixed(1)}s / 3.0s):</strong>
                        <div className="progress-bar">
                          <div 
                            className="progress-fill" 
                            style={{ width: `${Math.min((holdDuration / 3.0) * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {mousePos && (
                      <div className="prediction-confidence mouse-control">
                        <strong>🖱️ Mouse:</strong> ({mousePos.x}, {mousePos.y})
                      </div>
                    )}
                    {!lastPredictionAt && (
                      <div className="prediction-confidence">
                        Waiting for first prediction...
                      </div>
                    )}
                  </div>
                )}

                <div className="button-row">
                  {!liveStream ? (
                    <button className="primary" onClick={startControlStream}>
                      Start camera
                    </button>
                  ) : (
                    <button className="primary" onClick={stopControlStream}>
                      Stop camera
                    </button>
                  )}
                </div>
              </div>
            )}

            {active === 'gestures' && (
              <div className="panel-body">
                <h2>Gesture Library</h2>
                <p>
                  Add or remove gestures and map each one to a desktop action. Updates are
                  reflected instantly in your control flow.
                </p>
                <div className="gesture-grid">
                  <div className="gesture-panel">
                    <h3>Existing gestures</h3>
                    {configError && (
                      <div className="status-empty">{configError}</div>
                    )}
                    <div className="gesture-list">
                      {gestures.map((gesture) => (
                        <div key={gesture} className="gesture-item">
                          <span>{gesture}</span>
                          <button
                            className="ghost"
                            type="button"
                            onClick={() => deleteGesture(gesture)}
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                      {gestures.length === 0 && (
                        <div className="status-empty">No gestures yet.</div>
                      )}
                    </div>
                  </div>
                  <div className="gesture-panel">
                    <h3>Add a gesture</h3>
                    <label className="field">
                      Gesture name
                      <input
                        type="text"
                        value={newGesture}
                        onChange={(event) => setNewGesture(event.target.value)}
                        placeholder="e.g. swipe"
                      />
                    </label>
                    <button className="primary" type="button" onClick={addGesture}>
                      Add gesture
                    </button>
                    <button className="ghost" type="button" onClick={loadConfig}>
                      Reload gestures
                    </button>
                  </div>
                </div>

                <div className="mapping-panel">
                  <h3>Action mapping</h3>
                  <div className="mapping-list">
                    {gestures.map((gesture) => (
                      <div key={gesture} className="mapping-row">
                        <span>{gesture}</span>
                        <select
                          value={mappings[gesture] || 'none'}
                          onChange={(event) =>
                            updateMapping(gesture, event.target.value)
                          }
                        >
                          {Object.entries(actions).map(([key, label]) => (
                            <option key={key} value={key}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                    {gestures.length === 0 && (
                      <div className="status-empty">Add gestures to configure actions.</div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {active === 'dataset' && (
              <div className="panel-body">
                <h2>Make dataset</h2>
                <p>
                  Capture labeled samples directly from the webcam. Choose a gesture, start
                  the camera, and collect frames for training.
                </p>
                <div className="dataset-add">
                  <label className="field">
                    New label
                    <input
                      type="text"
                      value={datasetNewGesture}
                      onChange={(event) => setDatasetNewGesture(event.target.value)}
                      placeholder="e.g. pinch"
                    />
                  </label>
                  <button className="ghost" type="button" onClick={addDatasetGesture}>
                    Add label
                  </button>
                </div>
                <label className="field">
                  Gesture label
                  <select
                    value={datasetPose}
                    onChange={(event) => setDatasetPose(event.target.value)}
                    disabled={gestures.length === 0}
                  >
                    {gestures.length === 0 && (
                      <option value="">No gestures loaded</option>
                    )}
                    {gestures.map((gesture) => (
                      <option key={gesture} value={gesture}>
                        {gesture}
                      </option>
                    ))}
                  </select>
                </label>
                <button className="ghost" type="button" onClick={loadConfig}>
                  Reload gestures
                </button>

                <div className="video-container">
                  <canvas ref={datasetCanvasRef} style={{ display: 'none' }} />
                  <video
                    ref={datasetVideoRef}
                    autoPlay
                    playsInline
                    className="video-layer"
                  />
                </div>

                <div className="dataset-meta">
                  <span>Captured for {datasetPose || 'gesture'}:</span>
                  <strong>{datasetCount}</strong>
                </div>

                <div className="button-row">
                  {!datasetLive ? (
                    <button className="ghost" type="button" onClick={startDatasetCamera}>
                      Start camera
                    </button>
                  ) : (
                    <button className="ghost" type="button" onClick={stopDatasetCamera}>
                      Stop camera
                    </button>
                  )}
                  {!datasetRunning ? (
                    <button className="primary" type="button" onClick={startDatasetCapture}>
                      Start capture
                    </button>
                  ) : (
                    <button className="primary" type="button" onClick={stopDatasetCapture}>
                      Stop capture
                    </button>
                  )}
                  <button className="ghost" type="button" onClick={clearDatasetSamples}>
                    Clear samples
                  </button>
                  <button
                    className="ghost"
                    type="button"
                    onClick={() => deleteGesture(datasetPose)}
                    disabled={!datasetPose}
                  >
                    Delete label
                  </button>
                </div>
              </div>
            )}

            {active === 'model' && (
              <div className="panel-body">
                <h2>Retrain the model</h2>
                <p>
                  After updating gestures or adding new samples, retrain the model in one
                  click. The system will refresh live predictions automatically.
                </p>
                <div className="retrain-card">
                  <div>
                    <span>Status</span>
                    <strong>{retrainStatus}</strong>
                  </div>
                  <button
                    className="primary"
                    type="button"
                    onClick={retrainModel}
                    disabled={retrainRunning}
                  >
                    {retrainRunning ? 'Retraining...' : 'Retrain now'}
                  </button>
                </div>
              </div>
            )}
          </div>

          <aside className="panel status-panel">
            <div className="panel-body">
              <h3>System status</h3>
              <p>Latest system updates and retraining feedback.</p>
              <div className="status-list">
                {statusItems.length === 0 && (
                  <div className="status-empty">No activity yet. Try a workflow.</div>
                )}
                {statusItems.map((item) => (
                  <div key={item.id} className={`status-item ${item.kind}`}>
                    <div>
                      <strong>{item.label}</strong>
                      <span>{item.timestamp}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </div>
      </section>
    </div>
  );
}

export default App;
