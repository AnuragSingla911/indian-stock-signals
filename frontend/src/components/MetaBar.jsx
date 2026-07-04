import PropTypes from 'prop-types';

function formatIST(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const parts = new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata',
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(d);
  return `${parts} IST`;
}

export default function MetaBar({ meta }) {
  const date = formatIST(meta.generated_at);
  const model = meta.model_trained
    ? meta.model_name || 'trained'
    : 'heuristic fallback';
  return (
    <div className="meta-bar">
      <span>Data as of: <strong>{date}</strong></span>
      <span>Horizon: <strong>{meta.horizon_days} trading days</strong></span>
      <span>Universe: <strong>{meta.universe_size} stocks</strong></span>
      <span>ML model: <strong>{model}</strong></span>
    </div>
  );
}

MetaBar.propTypes = {
  meta: PropTypes.shape({
    generated_at: PropTypes.string,
    horizon_days: PropTypes.number,
    universe_size: PropTypes.number,
    model_trained: PropTypes.bool,
    model_name: PropTypes.string,
  }).isRequired,
};
