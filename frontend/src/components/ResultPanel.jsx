const LOW_THRESHOLD = 0.50;
const HIGH_THRESHOLD = 0.80;

function formatLabel(raw) {
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

function getConfidenceBand(prob) {
  if (prob >= HIGH_THRESHOLD) return "high";
  if (prob >= LOW_THRESHOLD) return "medium";
  return "low";
}

const BAND_CONFIG = {
  high: {
    label: "High",
    accentColor: "#4ade80",
    bgTint: "rgba(74,222,128,0.08)",
    borderTint: "rgba(74,222,128,0.25)",
    banner: "High confidence. No retake needed.",
    bannerIcon: "\u2705",
    outcome: "Result is stable \u2014 no retake needed.",
  },
  medium: {
    label: "Medium",
    accentColor: "#fbbf24",
    bgTint: "rgba(251,191,36,0.08)",
    borderTint: "rgba(251,191,36,0.25)",
    banner: "Medium confidence. Retake recommended.",
    bannerIcon: "\u26A0\uFE0F",
    outcome: "Try a closer single-leaf photo in better lighting.",
  },
  low: {
    label: "Low",
    accentColor: "#f87171",
    bgTint: "rgba(248,113,113,0.08)",
    borderTint: "rgba(248,113,113,0.25)",
    banner: "Low confidence. Retake required; check Top-3.",
    bannerIcon: "\u2757",
    outcome: null,
    tips: [
      "Use a single leaf and fill the frame",
      "Good lighting, avoid blur",
      "Try 2\u20133 photos and compare Top-3",
    ],
  },
};

function ConfidenceBar({ prob }) {
  const pct = (prob * 100).toFixed(0);
  const band = getConfidenceBand(prob);
  const color = BAND_CONFIG[band].accentColor;
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

function ConfidenceBadge({ band }) {
  const config = BAND_CONFIG[band];
  return (
    <span
      className="confidence-badge"
      style={{
        backgroundColor: config.bgTint,
        color: config.accentColor,
        borderColor: config.borderTint,
      }}
    >
      {config.label}
    </span>
  );
}

function GuidanceBanner({ band }) {
  const config = BAND_CONFIG[band];
  return (
    <div
      className="guidance-banner"
      style={{
        backgroundColor: config.bgTint,
        borderColor: config.borderTint,
      }}
    >
      <span style={{ color: config.accentColor }}>{config.banner}</span>
    </div>
  );
}

function AnalysisOutcome({ result, band, onUploadAnother, onClear }) {
  const config = BAND_CONFIG[band];
  return (
    <div className="outcome-panel">
      <div className="outcome-header">
        <h3>Analysis Outcome</h3>
        <ConfidenceBadge band={band} />
      </div>

      <p className="outcome-summary">
        Most likely: <strong>{formatLabel(result.top1.label)}</strong>
      </p>

      {config.outcome && (
        <p className="outcome-guidance">{config.outcome}</p>
      )}

      {band === "low" && config.tips && (
        <ul className="outcome-tips">
          {config.tips.map((tip, i) => (
            <li key={i}>{tip}</li>
          ))}
        </ul>
      )}

      <div className="outcome-actions">
        <button className="btn btn-accent btn-sm" onClick={onUploadAnother}>
          Upload Another Image
        </button>
        <button className="btn btn-ghost btn-sm" onClick={onClear}>
          Clear
        </button>
      </div>
    </div>
  );
}

function ThresholdLegend() {
  const lowPct = (LOW_THRESHOLD * 100).toFixed(0);
  const highPct = (HIGH_THRESHOLD * 100).toFixed(0);
  return (
    <div className="threshold-legend">
      <span className="threshold-title">Confidence Thresholds</span>
      <div className="threshold-rows">
        <div className="threshold-row">
          <span className="threshold-dot" style={{ background: "#f87171" }} />
          <span className="threshold-label">Low</span>
          <span className="threshold-range">Less than {lowPct}%</span>
        </div>
        <div className="threshold-row">
          <span className="threshold-dot" style={{ background: "#fbbf24" }} />
          <span className="threshold-label">Medium</span>
          <span className="threshold-range">{lowPct}% to {highPct}%</span>
        </div>
        <div className="threshold-row">
          <span className="threshold-dot" style={{ background: "#4ade80" }} />
          <span className="threshold-label">High</span>
          <span className="threshold-range">Greater than {highPct}%</span>
        </div>
      </div>
    </div>
  );
}

function ResultPanel({ result, loading, error, onClear, onUploadAnother }) {
  if (error) {
    return (
      <section className="glass-card result-panel">
        <h2 className="panel-title">Diagnosis Result</h2>
        <div className="error-box">{error}</div>
      </section>
    );
  }

  if (loading) {
    return (
      <section className="glass-card result-panel">
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
      <section className="glass-card result-panel">
        <h2 className="panel-title">Diagnosis Result</h2>
        <p className="placeholder-text">Upload an image to see results.</p>
      </section>
    );
  }

  const band = getConfidenceBand(result.top1.prob);

  return (
    <section className="glass-card result-panel">
      <h2 className="panel-title">Diagnosis Result</h2>

      <div className="top1-section">
        <span className="top1-subtitle">Detected class</span>
        <div className="top1-row">
          <span className="top1-label">{formatLabel(result.top1.label)}</span>
          <span className="top1-prob" style={{ color: BAND_CONFIG[band].accentColor }}>
            {(result.top1.prob * 100).toFixed(0)}%
          </span>
        </div>
        <ConfidenceBar prob={result.top1.prob} />
      </div>

      <div className="topk-section">
        <h3>Prediction - Top {result.topk.length}</h3>
        <table className="topk-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Class</th>
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

      <GuidanceBanner band={band} />

      <AnalysisOutcome
        result={result}
        band={band}
        onUploadAnother={onUploadAnother}
        onClear={onClear}
      />

      <ThresholdLegend />
    </section>
  );
}

export default ResultPanel;
