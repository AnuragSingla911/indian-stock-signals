import PropTypes from 'prop-types';

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function sentimentLabel(value) {
  if (value >= 0.3) return 'positive';
  if (value <= -0.3) return 'negative';
  return 'neutral';
}

export default function InsightsPanel({ insights }) {
  if (!insights) return null;

  const { macro, stock_highlights: highlights } = insights;

  return (
    <section className="insights-panel" aria-label="Market insights">
      <h2>Market context &amp; news</h2>
      {macro && (
        <div className="macro-bar">
          <p className="macro-summary">{macro.summary}</p>
          <div className="macro-stats">
            <span title="Nifty 50 one-month return">
              Nifty 1M: <strong>{pct(macro.nifty_1m_return)}</strong>
            </span>
            <span title="USD/INR one-month change">
              USD/INR 1M: <strong>{pct(macro.usd_inr_1m_change)}</strong>
            </span>
            <span title="Brent crude one-month return">
              Brent 1M: <strong>{pct(macro.brent_1m_return)}</strong>
            </span>
          </div>
        </div>
      )}
      {highlights && highlights.length > 0 && (
        <ul className="headline-list">
          {highlights.map((h) => (
            <li key={`${h.symbol}-${h.headline}`} className="headline-item">
              <span className={`sentiment-tag ${sentimentLabel(h.sentiment)}`}>
                {h.symbol}
              </span>
              {h.url ? (
                <a href={h.url} target="_blank" rel="noopener noreferrer">
                  {h.headline}
                </a>
              ) : (
                <span>{h.headline}</span>
              )}
              {h.publisher && <span className="publisher"> — {h.publisher}</span>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

InsightsPanel.propTypes = {
  insights: PropTypes.shape({
    macro: PropTypes.shape({
      summary: PropTypes.string,
      nifty_1m_return: PropTypes.number,
      usd_inr_1m_change: PropTypes.number,
      brent_1m_return: PropTypes.number,
    }),
    stock_highlights: PropTypes.arrayOf(
      PropTypes.shape({
        symbol: PropTypes.string,
        headline: PropTypes.string,
        sentiment: PropTypes.number,
        publisher: PropTypes.string,
        url: PropTypes.string,
      })
    ),
  }),
};
