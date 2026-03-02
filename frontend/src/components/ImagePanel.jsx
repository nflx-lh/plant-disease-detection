import { useRef } from "react";

function ImagePanel({ previewUrl, onFileSelect }) {
  const fileInput = useRef(null);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  };

  return (
    <section className="panel image-panel">
      <h2 className="panel-title">Image for Analysis</h2>

      <div className="preview-area">
        {previewUrl ? (
          <img src={previewUrl} alt="Selected leaf" className="preview-image" />
        ) : (
          <div className="preview-placeholder">
            <svg
              width="64"
              height="64"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#bbb"
              strokeWidth="1.5"
            >
              <rect x="2" y="4" width="20" height="16" rx="2" />
              <circle cx="12" cy="12" r="3" />
              <path d="M8 4V2h8v2" />
            </svg>
          </div>
        )}
      </div>

      <div className="button-group">
        <button
          className="btn btn-secondary"
          disabled
          title="Coming in Phase 2"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 15.2a3.2 3.2 0 100-6.4 3.2 3.2 0 000 6.4z" />
            <path d="M9 2L7.17 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V6a2 2 0 00-2-2h-3.17L15 2H9z" />
          </svg>
          Take Photo
        </button>
        <button
          className="btn btn-primary"
          onClick={() => fileInput.current.click()}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" />
            <path d="M14 2v6h6M12 18v-6M9 15h6" />
          </svg>
          Upload Image
        </button>
      </div>

      <input
        ref={fileInput}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: "none" }}
        onChange={handleChange}
      />

      <div className="tips">
        <p>For best results:</p>
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
