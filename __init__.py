"""
Tesla Stock Prediction Models Package
Contains 5 AI models for stock price prediction
"""

from .lstm_model import LSTMStockPredictor
from .gru_model import GRUStockPredictor
from .transformer_model import TransformerStockPredictor
from .xgboost_model import XGBoostStockPredictor
from .prophet_model import ProphetStockPredictor

__all__ = [
    'LSTMStockPredictor',
    'GRUStockPredictor',
    'TransformerStockPredictor',
    'XGBoostStockPredictor',
    'ProphetStockPredictor'
]

__version__ = '1.0.0'
