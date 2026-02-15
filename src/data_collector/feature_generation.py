from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

LOG = logging.getLogger(__name__)

class FeatureGeneratorBase(ABC):
    def __init__(self, config: Dict):
        self.config = config
        
    @abstractmethod
    def get_target(self, df_features: pd.DataFrame, window: int=10, h: int=10) -> List[float]:
        window = 10
        h = 10

        mid_prices = np.array(df_features["mid_price"])

        y_list = []

        for t in tqdm(range(len(mid_prices))):

            if t <= 2:
                y_list.append(0)
                continue
            
            if (t + window + h - 5) > len(df_features):
                y_list.append(0)
                continue

            p_t = mid_prices[t]
            m_prev = np.mean(mid_prices[max(0, (t-window)):t])
            m_next = np.mean(mid_prices[t+h:t+h+window])

            bps = p_t * 1e-4
            
            
            delta = m_next - m_prev

            if -bps <= delta and delta <= bps:
                y_list.append(0)

            
            if (-bps * 4) <= delta and delta < -bps:
                y_list.append(1)
            
            if delta < (-bps * 4):
                y_list.append(3)
                
            
            if (bps * 4) >= delta and delta > bps:
                y_list.append(2)
                
            if delta > (bps * 4):
                y_list.append(4)

        return y_list
    
    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class FeatureGenerator(FeatureGeneratorBase):
    def __init__(self, config):
        super().__init__(config)
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # generate general features
        data['bid_ask_spread'] = data['ask_price_1'] - data['best_bid_price_1']
        data["mid_price"] = (data['ask_price_1'] + data['best_bid_price_1']) / 2
        
        #ToDO: add more features here (e.g., order book imbalance, volume features, etc.)
        return data
        