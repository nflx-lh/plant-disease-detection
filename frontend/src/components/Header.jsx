function Header({ modelVersion }) {
  return (
    <header className="header">
      <div className="header-left">
        <span className="header-icon">&#127807;</span>
        <h1>AnovaGreen Field Diagnosis</h1>
      </div>
      <span className="header-meta">
        {modelVersion ? "Model v1.0 \u00B7 AnovaGreen" : "Connecting\u2026"}
      </span>
    </header>
  );
}

export default Header;
