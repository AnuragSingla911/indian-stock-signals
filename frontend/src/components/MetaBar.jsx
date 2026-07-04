import PropTypes from 'prop-types';

export default function MetaBar({ meta }) {
  const date = meta.generated_at ? meta.generated_at.replace('T', ' ').replace('Z', ' UTC') : '—';
  return (
    <div className="meta-bar">
      <span>Data as of: <strong>{date}</strong></span>
      <span>Horizon: <strong>{meta.horizon_days} trading days</strong></span>
      <span>Universe: <strong>{meta.universe_size} stocks</strong></span>
      <span>ML model: <strong>{meta.model_trained ? 'trained' : 'heuristic fallback'}</strong></span>
    </div>
  );
}

MetaBar.propTypes = {
  meta: PropTypes.shape({
    generated_at: PropTypes.string,
    horizon_days: PropTypes.number,
    universe_size: PropTypes.number,
    model_trained: PropTypes.bool,
  }).isRequired,
};
