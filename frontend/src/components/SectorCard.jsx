import PropTypes from 'prop-types';
import StockRow from './StockRow.jsx';
import ScoreBadge from './ScoreBadge.jsx';

export default function SectorCard({ sector }) {
  return (
    <section className="sector-card">
      <header className="sector-head">
        <h2>{sector.display_name}</h2>
        <ScoreBadge score={sector.sector_score} label="Sector score (0-100)" />
      </header>
      <p className="sector-rationale">{sector.sector_rationale}</p>
      <ol className="stock-list">
        {sector.stocks.map((stock, i) => (
          <StockRow key={stock.symbol} stock={stock} rank={i + 1} />
        ))}
      </ol>
    </section>
  );
}

SectorCard.propTypes = {
  sector: PropTypes.shape({
    display_name: PropTypes.string.isRequired,
    sector_score: PropTypes.number.isRequired,
    sector_rationale: PropTypes.string.isRequired,
    stocks: PropTypes.array.isRequired,
  }).isRequired,
};
