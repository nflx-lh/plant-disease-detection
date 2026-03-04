import { useState, useEffect, useRef, useCallback } from "react";

function CameraModal({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 960 } } })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(() => onClose());

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [onClose]);

  const handleCapture = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
          onCapture(file);
        }
      },
      "image/jpeg",
      0.92
    );
  };

  return (
    <div className="camera-overlay" onClick={onClose}>
      <div className="camera-modal" onClick={(e) => e.stopPropagation()}>
        <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
        <div className="camera-controls">
          <button className="btn btn-ghost btn-sm" onClick={onClose}>Cancel</button>
          <button className="camera-shutter" onClick={handleCapture} title="Capture" />
          <div style={{ width: 64 }} />
        </div>
      </div>
    </div>
  );
}

function ImagePanel({ previewUrl, onFileSelect, fileInputRef }) {
  const [cameraSupported, setCameraSupported] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("checking"); // "checking" | "supported" | "unsupported" | "denied"
  const [showCamera, setShowCamera] = useState(false);
  const cameraInputRef = useRef(null);

  useEffect(() => {
    const isSecure = window.isSecureContext === true || window.location.hostname === "localhost";
    const hasMediaDevices = !!navigator.mediaDevices?.getUserMedia;

    if (!isSecure || !hasMediaDevices) {
      setCameraSupported(false);
      setCameraStatus("unsupported");
      return;
    }

    // Check permissions if the API is available
    if (navigator.permissions?.query) {
      navigator.permissions.query({ name: "camera" }).then((permStatus) => {
        if (permStatus.state === "denied") {
          setCameraSupported(false);
          setCameraStatus("denied");
        } else {
          setCameraSupported(true);
          setCameraStatus("supported");
        }
        permStatus.onchange = () => {
          if (permStatus.state === "denied") {
            setCameraSupported(false);
            setCameraStatus("denied");
          } else {
            setCameraSupported(true);
            setCameraStatus("supported");
          }
        };
      }).catch(() => {
        // permissions API not supported for camera, assume available
        setCameraSupported(true);
        setCameraStatus("supported");
      });
    } else {
      setCameraSupported(true);
      setCameraStatus("supported");
    }
  }, []);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
    e.target.value = "";
  };

  const handleCameraClick = () => {
    // On mobile, use the native capture input (Option A) for best UX
    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    if (isMobile && cameraInputRef.current) {
      cameraInputRef.current.click();
    } else {
      setShowCamera(true);
    }
  };

  const handleCameraCapture = useCallback((file) => {
    setShowCamera(false);
    onFileSelect(file);
  }, [onFileSelect]);

  const handleCameraClose = useCallback(() => {
    setShowCamera(false);
  }, []);

  const cameraButtonLabel = cameraStatus === "denied"
    ? "Camera permission denied"
    : !cameraSupported
      ? "Camera not available"
      : "Take Photo";

  const cameraHelperText = cameraStatus === "denied"
    ? "Camera permission denied \u2014 use Upload Image."
    : !cameraSupported
      ? "Camera not available in this browser. Use Upload Image."
      : null;

  return (
    <section className="glass-card image-panel">
      <h2 className="panel-title">Image for Analysis</h2>

      <div className="preview-area">
        {previewUrl ? (
          <img src={previewUrl} alt="Selected leaf" className="preview-image" />
        ) : (
          <div className="preview-placeholder">
            <svg
              width="56"
              height="56"
              viewBox="0 0 24 24"
              fill="none"
              stroke="rgba(100,116,139,0.5)"
              strokeWidth="1.5"
            >
              <rect x="2" y="4" width="20" height="16" rx="2" />
              <circle cx="12" cy="12" r="3" />
              <path d="M8 4V2h8v2" />
            </svg>
            <span className="preview-label">No image selected</span>
          </div>
        )}
      </div>

      <div className="button-group">
        <button
          className="btn btn-orange"
          disabled={!cameraSupported}
          title={cameraSupported ? "Take a photo with your camera" : cameraHelperText}
          onClick={handleCameraClick}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 15.2a3.2 3.2 0 100-6.4 3.2 3.2 0 000 6.4z" />
            <path d="M9 2L7.17 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V6a2 2 0 00-2-2h-3.17L15 2H9z" />
          </svg>
          {cameraButtonLabel}
        </button>
        <button
          className="btn btn-accent"
          onClick={() => fileInputRef.current.click()}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
          </svg>
          Upload Image
        </button>
      </div>

      {cameraHelperText && (
        <p className="camera-helper-text">{cameraHelperText}</p>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: "none" }}
        onChange={handleChange}
      />

      {/* Hidden native camera input for mobile (Option A) */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: "none" }}
        onChange={handleChange}
      />

      {showCamera && (
        <CameraModal onCapture={handleCameraCapture} onClose={handleCameraClose} />
      )}

      <div className="tips-card">
        <p className="tips-heading">For best results</p>
        <ul>
          <li>Single leaf, centered in frame</li>
          <li>Good lighting, avoid blur</li>
          <li>Fill frame with the leaf</li>
        </ul>
      </div>
    </section>
  );
}

export default ImagePanel;
