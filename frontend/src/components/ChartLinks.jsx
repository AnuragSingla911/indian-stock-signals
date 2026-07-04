import PropTypes from 'prop-types';

export default function ChartLinks({ links, symbol }) {
  return (
    <span className="chart-links">
      <a href={links.tradingview} target="_blank" rel="noopener noreferrer">
        TradingView ↗
      </a>
      <a href={links.yahoo} target="_blank" rel="noopener noreferrer">
        Yahoo ↗
      </a>
      <span className="sr-only">charts for {symbol}</span>
    </span>
  );
}

ChartLinks.propTypes = {
  links: PropTypes.shape({
    tradingview: PropTypes.string.isRequired,
    yahoo: PropTypes.string.isRequired,
  }).isRequired,
  symbol: PropTypes.string.isRequired,
};
