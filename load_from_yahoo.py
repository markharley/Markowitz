import pandas as pd
from zipline.utils.factory import load_bars_from_yahoo
import matplotlib.pyplot as plt

end = pd.Timestamp.utcnow()
start = end - 2500 * pd.tseries.offsets.BDay()

stocks = ['IBM', 'GLD', 'XOM', 'AAPL', 'MSFT', 'SHY', 'GOOG']

data = load_bars_from_yahoo(stocks=stocks, start=start, end=end)

data.loc[:, :, 'price'].plot(figsize=(8,5))
plt.ylabel('price in $')
plt.show()