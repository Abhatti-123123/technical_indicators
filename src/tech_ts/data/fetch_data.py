import yfinance as yf
import pandas as pd
import unittest

from tech_ts.data.constants import TICKERS
from tech_ts.data.date_config import START_DATE, END_DATE

def download_prices(tickers=TICKERS, start=START_DATE, end=END_DATE):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=False, progress=False)['Close']
    data.dropna(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)
    return data

class TestFetchData(unittest.TestCase):
    def setUp(self):
        self.prices = download_prices()

    def test_download_non_empty(self):
        self.assertFalse(self.prices.empty, "Downloaded data is empty")

    def test_columns_present(self):
        self.assertTrue(all(ticker in self.prices.columns for ticker in TICKERS),
                        "Not all tickers are present in the columns")

    def test_index_type(self):
        self.assertIsInstance(self.prices.index, pd.DatetimeIndex,
                              "Index is not DatetimeIndex")

    def test_no_missing_values(self):
        self.assertFalse(self.prices.isnull().values.any(), "Data contains NaN values")

if __name__ == "__main__":
    unittest.main(verbosity=2)
