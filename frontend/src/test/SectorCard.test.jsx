import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import SectorCard from '../components/SectorCard.jsx';

const sector = {
  sector: 'IT',
  display_name: 'Information Technology',
  sector_score: 78.4,
  sector_rationale: 'Strong 3M sector momentum; broad participation.',
  stocks: [
    {
      symbol: 'TCS',
      name: 'Tata Consultancy Services',
      composite_score: 82.1,
      up_probability: 0.61,
      rationale: 'Ranks highly on strong momentum and quality.',
      chart_links: {
        tradingview: 'https://www.tradingview.com/chart/?symbol=NSE:TCS',
        yahoo: 'https://finance.yahoo.com/quote/TCS.NS',
      },
    },
  ],
};

describe('SectorCard', () => {
  it('renders sector name, score and stock with chart links', () => {
    render(<SectorCard sector={sector} />);
    expect(screen.getByText('Information Technology')).toBeInTheDocument();
    expect(screen.getByText('TCS')).toBeInTheDocument();
    expect(screen.getByText(/Ranks highly/)).toBeInTheDocument();

    const tv = screen.getByText(/TradingView/);
    expect(tv).toHaveAttribute('href', 'https://www.tradingview.com/chart/?symbol=NSE:TCS');
    expect(tv).toHaveAttribute('target', '_blank');
    expect(tv).toHaveAttribute('rel', 'noopener noreferrer');

    expect(screen.getByText('Up-prob: 61%')).toBeInTheDocument();
  });
});
