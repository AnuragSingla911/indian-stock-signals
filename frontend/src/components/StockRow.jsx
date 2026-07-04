import PropTypes from 'prop-types';
import ScoreBadge from './ScoreBadge.jsx';
import ChartLinks from './ChartLinks.jsx';

export default function StockRow({ stock, rank }) {
  return (
    <li className="stock-row">
      <div className="stock-head">
        <span className="rank">#{rank}</span>
        <span className="symbol">{stock.symbol}</span>
        <span className="company">{stock.name}</span>
        <ScoreBadge score={stock.composite_score} />
      </div>
      <p className="rationale">{stock.rationale}</p>
      <div className="stock-foot">
        <span className="prob" title="Model-estimated probability of positive forward return">
          Up-prob: {Math.round(stock.up_probability * 100)}%
        </span>
        <ChartLinks links={stock.chart_links} symbol={stock.symbol} />
      </div>
    </li>
  );
}

StockRow.propTypes = {
  stock: PropTypes.shape({
    symbol: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    composite_score: PropTypes.number.isRequired,
    up_probability: PropTypes.number.isRequired,
    rationale: PropTypes.string.isRequired,
    chart_links: PropTypes.object.isRequired,
  }).isRequired,
  rank: PropTypes.number.isRequired,
};
