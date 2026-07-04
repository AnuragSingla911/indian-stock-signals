import PropTypes from 'prop-types';

export default function Disclaimer({ text }) {
  return (
    <div className="disclaimer" role="note">
      <strong>⚠ Not investment advice.</strong> {text}
    </div>
  );
}

Disclaimer.propTypes = {
  text: PropTypes.string.isRequired,
};
