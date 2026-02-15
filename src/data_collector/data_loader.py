from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd


LOG = logging.getLogger(__name__)


@dataclass
class BaseLoadConfig():
    """Abstract dataclass for data-loading configuration.

    Fields:
      - path_to_orderbook: directory where per-date parquet orderbooks live
      - path_to_trades: (optional) directory or file path for trades data
      - dates_orderbook: list of date strings (without extension) for orderbook files
      - data_trades: explicit file path to the single trades parquet (if provided)
      - load_orderbook: whether to load orderbooks
      - load_trades: whether to load trades
      - cols_orderbook / cols_trades: list of columns to keep (None means all)
    """

    path_to_orderbook: str
    path_to_trades: Optional[str] = None
    dates_orderbook: Optional[List[str]] = None
    data_trades: Optional[str] = None
    load_orderbook: bool = True
    load_trades: bool = True
    cols_orderbook: Optional[List[str]] = None
    cols_trades: Optional[List[str]] = None

    @abstractmethod
    def validate(self) -> None:
        """Validate config. Concrete subclasses should implement checks and raise on invalid config."""
        raise NotImplementedError


@dataclass
class DataLoadConfig(BaseLoadConfig):
    """Concrete config with a basic validate implementation."""

    def validate(self) -> None:
        if not (self.load_orderbook or self.load_trades):
            raise ValueError("At least one of load_orderbook or load_trades must be True")

        if self.load_orderbook and (not self.dates_orderbook or len(self.dates_orderbook) == 0):
            raise ValueError("dates_orderbook must be provided when load_orderbook is True")

        # If a single trades file isn't provided, allow path_to_trades to point to a file
        if self.load_trades and (not self.data_trades and not self.path_to_trades):
            raise ValueError("Provide data_trades (file) or path_to_trades when load_trades is True")


class DataLoader(ABC):
    """Loader that loads orderbook(s) and/or trades according to a config.

    Usage:
      cfg = DataLoadConfig(...)
      loader = SomeConcreteLoader(cfg)
      orderbooks, trades = loader.load_data()

    Returns:
      - orderbooks: Optional[Dict[date_str, pd.DataFrame]] (None if not requested)
      - trades: Optional[pd.DataFrame] (None if not requested)
    """

    def __init__(self, cfg: BaseLoadConfig):
        self.cfg = cfg
        self.cfg.validate()

    @abstractmethod
    def load_data(self) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[pd.DataFrame]]:
        """Load data according to the config and return (orderbooks, trades)."""


class ParquetDataLoader(DataLoader):
    """Concrete loader that reads parquet files from disk."""

    def _orderbook_path_for_date(self, date: str) -> str:
        base = self.cfg.path_to_orderbook
        # allow both dir/ and full path cases
        if os.path.isdir(base):
            return os.path.join(base, f"{date}.parquet")
        # if base looks like a file path pattern, try to join anyway
        return os.path.join(base, f"{date}.parquet")

    def _read_parquet_safe(self, path: str, cols_needed: Optional[List[str]]) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet file not found: {path}")
        df = pd.read_parquet(path, columns=cols_needed)
        return df


    def load_orderbooks(self) -> Dict[str, pd.DataFrame]:
        """Load orderbook parquet files for dates listed in the config."""
        if not self.cfg.dates_orderbook:
            return {}
        out = []
        for date in self.cfg.dates_orderbook:
            path = self._orderbook_path_for_date(date)
            df = self._read_parquet_safe(path, self.cfg.cols_orderbook)
            out.append(df)
        return pd.concat(out, ignore_index=True)

    def load_trades(self) -> pd.DataFrame:
        """Load trades parquet. Uses data_trades if provided, otherwise path_to_trades."""
        path = self.cfg.path_to_trades + "/" + self.cfg.data_trades
        if path is None:
            raise ValueError("No trades file path provided in config")
        df = self._read_parquet_safe(path, self.cfg.cols_trades)
        return df

    def match_trades_orderbooks(self, window_size: str = "5min") -> pd.DataFrame:
        """
        Matches orderbooks to trade features: cumulative and window-based aggregates (e.g., volume, ratio, avg price)
        without lookahead bias. Computes window stats up to the last preceding trade for each orderbook.

        Args:
            window_size (str): Window size for aggregates, e.g., "5min", "1h". Uses pd.Timedelta.

        Returns:
            pd.DataFrame: Merged orderbooks with attached trade features (window aggs + last trade details).
        """
        LOG.debug("match_trades_orderbooks called with window_size=%s", window_size)
        orderbooks = self.load_orderbooks()
        trades = self.load_trades()
        
        # Ensure sorted and unique index
        orderbooks = orderbooks.sort_values('datetime').reset_index(drop=True)
        trades = trades.sort_values('datetime').reset_index(drop=True)
        
        trades['price'] = pd.to_numeric(trades['price'], errors='coerce').astype('float64')
        trades['quantity'] = pd.to_numeric(trades['quantity'], errors='coerce').astype('float64')
        
        # Cumulative features (vectorized, no loc/reindex issues)
        trades["cum_trades_count"] = trades.index + 1
        trades["cum_volume"] = trades["quantity"].cumsum()
        is_buy = trades["is_buyer_maker"].astype(int)
        trades["cum_buyer_count"] = trades["is_buyer_maker"].astype(int).cumsum()  # Count of buyer-initiated
        trades["cum_weighted"] = (trades["price"] * trades["quantity"]).cumsum()  # For weighted avg
        trades["cum_volume_buyer"]  = (trades["quantity"] * is_buy).cumsum()
        trades["cum_volume_seller"] = (trades["quantity"] * (1 - is_buy)).cumsum()
        trades["ratio_buy"] = trades["cum_buyer_count"] / trades["cum_trades_count"]  # Cumulative ratio
        trades["avg_price"] = trades["cum_weighted"] / trades["cum_volume"]  # Cumulative VWAP-like

        # Window features: subtract prev cum at (datetime - window)
        trades["shifted_datetime"] = trades["datetime"] - pd.Timedelta(window_size)
        right_cols = [
            'datetime', 'cum_volume_buyer', 'cum_volume_seller', 'cum_volume',
            'cum_trades_count', 'cum_buyer_count', 'cum_weighted'
        ]

        # merge_asof to find previous cumulative values at (datetime - window_size)
        # we merge the shifted timestamps against the original trades and then
        # rename the resulting columns to have a _prev suffix (so subsequent
        # lookups like temp['cum_volume_prev'] exist)
        temp = pd.merge_asof(
            trades[['shifted_datetime']].rename(columns={'shifted_datetime': 'datetime'}),
            trades[[c for c in right_cols]],
            on='datetime',
            direction='backward'
        )

        # Rename previous columns to have _prev suffix
        prev_map = {c: f"{c}_prev" for c in right_cols if c != 'datetime'}
        temp = temp.rename(columns=prev_map)

        # Window calcs (use _prev columns produced above)
        trades['window_volume'] = trades['cum_volume'] - temp['cum_volume_prev'].fillna(0)
        trades['window_volume_buyer'] = trades['cum_volume_buyer'] - temp['cum_volume_buyer_prev'].fillna(0)
        trades['window_volume_seller'] = trades['cum_volume_seller'] - temp['cum_volume_seller_prev'].fillna(0)
        window_total_count = trades['cum_trades_count'] - temp['cum_trades_count_prev'].fillna(0)
        window_buyer_count = trades['cum_buyer_count'] - temp['cum_buyer_count_prev'].fillna(0)
        trades['window_ratio_buy'] = window_buyer_count / (window_total_count + 1e-6)
        window_weighted = trades['cum_weighted'] - temp['cum_weighted_prev'].fillna(0)
        trades['window_avg_price'] = window_weighted / (trades['window_volume'] + 1e-6)
        
        trade_feature_cols = [
            c for c in trades.columns
            if c not in ['datetime', 'trade_id', 'price', 'quantity', 'is_buyer_maker']
        ]

        trades[trade_feature_cols] = trades[trade_feature_cols].shift(1)
        
        # Merge: attach to orderbooks (last preceding trade's features)
        drop_cols = ['shifted_datetime', 'trade_id'] + [col for col in right_cols if col != 'datetime']
        trades_merged = trades.drop(columns=drop_cols)  # Drop cum/prev, keep window + original trade cols
        merged = pd.merge_asof(
            orderbooks,
            trades_merged,
            on='datetime',
            direction='backward',
            suffixes=('_orderbook', '_trade')
        )
        
        LOG.debug("Merged shape: %s", merged.shape)
        return merged
    
    def load_data(self) -> Optional[pd.DataFrame]:
        out_data = None

        if self.cfg.load_orderbook and self.cfg.load_trades:
            # Delegate matching to the template method which the user can override
            try:
                out_data = self.match_trades_orderbooks()
            except NotImplementedError:
                LOG.info("match_trades_orderbooks not implemented")
                
        elif self.cfg.load_orderbook:
            LOG.info("Loading orderbooks for %d dates", len(self.cfg.dates_orderbook or []))
            out_data = self.load_orderbooks()

        elif self.cfg.load_trades:
            LOG.info("Loading trades from %s", self.cfg.data_trades or self.cfg.path_to_trades)
            out_data = self.load_trades()


        return out_data

