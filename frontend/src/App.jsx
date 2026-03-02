import { useState, useEffect } from "react";
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

  return (
    <div className="app">
      <Header modelVersion={modelVersion} />
      <main className="main-content">
        <ImagePanel previewUrl={previewUrl} onFileSelect={handleFileSelect} />
        <ResultPanel result={result} loading={loading} error={error} />
      </main>
    </div>
  );
}

export default App;
