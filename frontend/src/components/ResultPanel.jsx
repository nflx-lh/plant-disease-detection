function formatLabel(raw) {
  // "tomato__early_blight" -> "Tomato - Early Blight"
  const parts = raw.split("__");
  const plant = parts[0];
  const disease = parts.slice(1).join(" ").replace(/_/g, " ");
  const capitalize = (s) =>
    s
      .split(" ")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
  return `${capitalize(plant)} - ${capitalize(disease)}`;
}

function ConfidenceBar({ prob }) {
  const pct = (prob * 100).toFixed(0);
  const color = prob >= 0.7 ? "#2e7d32" : prob >= 0.4 ? "#f9a825" : "#c62828";
  return (
    <div className="confidence-bar-container">
      <div className="confidence-bar-track">
        <div
          className="confidence-bar-fill"
          style={{ width: `${prob * 100}%`, backgroundColor: color }}
        />
      </div>
      <span className="confidence-value">{pct}%</span>
    </div>
  );
}

function ResultPanel({ result, loading, error }) {
  if (error) {
    return (
      <section className="panel result-panel">
        <h2 className="panel-title">Diagnosis Result</h2>
        <div className="error-box">{error}</div>
      </section>
    );
  }

  if (loading) {
    return (
      <section className="panel result-panel">
        <h2 className="panel-title">Diagnosis Result</h2>
        <div className="loading">
          <div className="spinner" />
          <span>Analyzing image...</span>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel result-panel">
        <h2 className="panel-title">Diagnosis Result</h2>
        <p className="placeholder-text">Upload an image to see results.</p>
      </section>
    );
  }

  return (
    <section className="panel result-panel">
      <h2 className="panel-title">Diagnosis Result</h2>

      <div className="top1-section">
        <span className="top1-subtitle">Most likely disease</span>
        <div className="top1-row">
          <span className="top1-label">{formatLabel(result.top1.label)}</span>
          <span className="top1-prob">{(result.top1.prob * 100).toFixed(0)}%</span>
        </div>
        <ConfidenceBar prob={result.top1.prob} />
      </div>

      {result.warning && (
        <div className="warning-box">&#9888; {result.warning}</div>
      )}

      <div className="topk-section">
        <h3>Top {result.topk.length} Prediction</h3>
        <table className="topk-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Disease Class</th>
              <th>Probability</th>
            </tr>
          </thead>
          <tbody>
            {result.topk.map((item) => (
              <tr key={item.rank}>
                <td>{item.rank}</td>
                <td>{formatLabel(item.label)}</td>
                <td>{(item.prob * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="caution-note">
        &#9888; Note: Field images are challenging; use caution and Top-3
        classes.
      </div>
    </section>
  );
}

export default ResultPanel;
