import PropTypes from 'prop-types';

function colorFor(score) {
  if (score >= 80) return '#0f9d58';
  if (score >= 60) return '#3ba55d';
  if (score >= 40) return '#d9a334';
  return '#c94b4b';
}

export default function ScoreBadge({ score, label }) {
  return (
    <span
      className="score-badge"
      style={{ backgroundColor: colorFor(score) }}
      title={label || 'Composite score (0-100)'}
    >
      {Math.round(score)}
    </span>
  );
}

ScoreBadge.propTypes = {
  score: PropTypes.number.isRequired,
  label: PropTypes.string,
};
