import { useState, useEffect, useRef } from "react";
import Header from "./components/Header";
import ImagePanel from "./components/ImagePanel";
import ResultPanel from "./components/ResultPanel";
import { checkHealth, predictImage } from "./api/predict";
import "./App.css";

function App() {
  const [modelVersion, setModelVersion] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    checkHealth()
      .then((data) => setModelVersion(data.model_version))
      .catch(() =>
        setError("Cannot connect to backend. Is it running on port 8000?")
      );
  }, []);

  const handleFileSelect = async (file) => {
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setLoading(false);
  };

  const handleUploadAnother = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="app">
      <Header modelVersion={modelVersion} />
      <main className="main-content">
        <ImagePanel
          previewUrl={previewUrl}
          onFileSelect={handleFileSelect}
          fileInputRef={fileInputRef}
        />
        <ResultPanel
          result={result}
          loading={loading}
          error={error}
          onClear={handleClear}
          onUploadAnother={handleUploadAnother}
        />
      </main>
    </div>
  );
}

export default App;
