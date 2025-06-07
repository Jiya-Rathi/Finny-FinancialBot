# ─── utils/prophet_forecast.py ─────────────────────────────────────────────────

import pandas as pd
from prophet import Prophet

class CashFlowForecaster:
    """
    Uses Prophet to forecast monthly or daily cash flow based on historical transactions.
    """
    def __init__(self):
        pass

    def prepare_dataframe(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform `transactions_df` into Prophet‐compatible DataFrame:
        - Columns: ds (date), y (net cash flow for that date)
        """
        # Aggregate daily net cash flow
        daily = transactions_df.groupby(transactions_df['Date'].dt.date)['Amount']\
                   .sum().reset_index()
        daily.columns = ['ds', 'y']
        daily['ds'] = pd.to_datetime(daily['ds'])
        return daily

    def forecast(self, transactions_df: pd.DataFrame, periods: int = None) -> dict:
        """
        Return a dict: { 'ds': [...dates...], 'yhat': [...forecasted_values...] }
        """
        if periods is None:
            from config.settings import PROPHET_DEFAULT_PERIODS
            periods = PROPHET_DEFAULT_PERIODS

        df_prophet = self.prepare_dataframe(transactions_df)
        m = Prophet()
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)

        # Return only the date and predicted yhat
        return {
            'ds': forecast['ds'].dt.date.tolist(),
            'yhat': forecast['yhat'].tolist()
        }
