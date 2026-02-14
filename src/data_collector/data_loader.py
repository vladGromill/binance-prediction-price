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

    def match_trades_orderbooks(self, orderbooks: Dict[str, pd.DataFrame], trades: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Template method to match trades with orderbooks by date.

        The user can override this method to implement custom matching logic.

        Parameters:
          - orderbooks: dict mapping date string -> orderbook DataFrame
          - trades: full trades DataFrame

        Returns:
          - (matched_orderbooks, matched_trades)

        Default implementation is a no-op and simply returns the inputs unchanged.
        """
        LOG.debug("Default match_trades_orderbooks called — returning inputs unchanged")
        return orderbooks, trades


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
        path = self.cfg.data_trades or self.cfg.path_to_trades
        if path is None:
            raise ValueError("No trades file path provided in config")
        df = self._read_parquet_safe(path, self.cfg.cols_trades)
        return df

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

