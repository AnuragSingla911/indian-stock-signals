import { useEffect, useState } from 'react';
import { fetchPredictions } from './api.js';
import Disclaimer from './components/Disclaimer.jsx';
import MetaBar from './components/MetaBar.jsx';
import SectorCard from './components/SectorCard.jsx';

const DEFAULT_DISCLAIMER =
  'Educational and informational use only. This is not investment advice. Markets are risky; ' +
  'model signals do not guarantee future results. Consult a SEBI-registered adviser.';

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    fetchPredictions()
      .then((d) => {
        if (active) setData(d);
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🇮🇳 Indian Stock Signals</h1>
        <p className="tagline">
          Top 5 sectors · 5 stocks each · transparent factor + ML ranking
        </p>
      </header>

      <Disclaimer text={data?.disclaimer || DEFAULT_DISCLAIMER} />

      {loading && <p className="status">Loading signals…</p>}

      {error && (
        <div className="error" role="alert">
          <p>Could not load signals: {error}</p>
          <p className="hint">
            Is the API running? Start it with <code>make api</code> and run{' '}
            <code>make pipeline</code> to generate predictions.
          </p>
        </div>
      )}

      {data && (
        <>
          <MetaBar meta={data} />
          <main className="sector-grid">
            {data.sectors.map((sector) => (
              <SectorCard key={sector.sector} sector={sector} />
            ))}
          </main>
          <footer className="app-footer">
            <p>
              Methodology: cross-sectional momentum, trend, quality, value and low-volatility
              factors blended with a gradient-boosted probability model. See the repo docs.
            </p>
          </footer>
        </>
      )}
    </div>
  );
}
