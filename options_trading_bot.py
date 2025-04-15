import json
import datetime
import pytz
import time
import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------------
# Technical Indicator Functions
# -------------------------------

def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(series, period=20):
    """Calculate the simple moving average (SMA)."""
    return series.rolling(window=period).mean()

def calculate_bollinger_bands(series, period=20, num_std=2):
    """Calculate Bollinger Bands (SMA, upper band, and lower band)."""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

# -------------------------------
# Options Trading Bot Class
# -------------------------------

class OptionsTradingBot:
    def __init__(self, config_file='config.json'):
        # Load tickers from a config file
        self.config = self.load_config(config_file)
        self.tickers = self.config.get("tickers", [])
        self.watchlist = []  # To store the trade suggestion and simulation details

        # Trade simulation parameters
        self.profit_target_multiplier = 1.20
        self.stop_loss_multiplier = 0.50
        self.assumed_delta = 0.5  # Simplified delta for simulation

        # Flag to ensure that only the trade at 10:00 is captured.
        self.trade_executed = False

    def load_config(self, filename):
        """Load configuration from a JSON file."""
        try:
            with open(filename, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config file {filename}: {e}")
            return {}

    def fetch_intraday_data(self, ticker, period='1d', interval='5m'):
        """
        Fetch intraday data using yfinance.
        Returns a DataFrame.
        """
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                print(f"No intraday data for {ticker}.")
            return data
        except Exception as e:
            print(f"Error fetching intraday data for {ticker}: {e}")
            return pd.DataFrame()

    def select_near_term_expiry(self, ticker):
        """
        For the given ticker, select an expiry date that is approximately 1 day-to-expiry.
        If an expiry exactly one day away is not available, choose the nearest future date.
        """
        ticker_obj = yf.Ticker(ticker)
        available_exps = ticker_obj.options
        if not available_exps:
            print(f"No options expiration data for {ticker}.")
            return None

        now = datetime.datetime.now(pytz.timezone('US/Eastern')).date()
        candidate = None
        min_diff = None
        for exp in available_exps:
            try:
                exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            except Exception as e:
                continue
            diff = (exp_date - now).days
            if diff >= 1:  # only consider future expiries
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    candidate = exp
        if candidate:
            return candidate
        else:
            print(f"No suitable expiry found for {ticker}.")
            return None

    def fetch_options_chain(self, ticker, expiry, option_type="call"):
        """
        Fetch the options chain for a given ticker and expiry date.
        Returns a DataFrame with either the calls or puts.
        """
        ticker_obj = yf.Ticker(ticker)
        try:
            chain = ticker_obj.option_chain(expiry)
            if option_type.lower() == "call":
                return chain.calls
            elif option_type.lower() == "put":
                return chain.puts
        except Exception as e:
            print(f"Error fetching options chain for {ticker}: {e}")
        return pd.DataFrame()

    def analyze_ticker(self, ticker):
        """
        Analyze the underlying ticker using intraday data to compute technical indicators.
        Generates a signal if:
          - Bullish (oversold): RSI < 30 and current price is near or below the lower Bollinger Band.
          - Bearish (overbought): RSI > 70 and current price is near or above the upper Bollinger Band.
        Returns a dictionary with the analysis details or None if conditions are not met.
        """
        data = self.fetch_intraday_data(ticker)
        if data.empty or len(data) < 20:
            return None

        # Calculate technical indicators
        data["RSI"] = calculate_rsi(data["Close"])
        sma, upper_band, lower_band = calculate_bollinger_bands(data["Close"])
        data["SMA20"] = sma
        data["UpperBand"] = upper_band
        data["LowerBand"] = lower_band

        latest = data.iloc[-1]
        close = latest["Close"]
        rsi = latest["RSI"]

        signal = None
        tolerance = 0.01  # 1% tolerance when comparing with the bands

        # Bullish signal: RSI < 30 and price near or below the lower Bollinger Band
        if rsi < 30 and close <= latest["LowerBand"] * (1 + tolerance):
            signal = "Bullish"
        # Bearish signal: RSI > 70 and price near or above the upper Bollinger Band
        elif rsi > 70 and close >= latest["UpperBand"] * (1 - tolerance):
            signal = "Bearish"

        analysis = {
            "ticker": ticker,
            "current_price": close,
            "RSI": round(rsi, 2),
            "SMA20": round(latest["SMA20"], 2),
            "UpperBand": round(latest["UpperBand"], 2),
            "LowerBand": round(latest["LowerBand"], 2),
            "signal": signal,
            "timestamp": latest.name  # timestamp of the latest data point
        }
        return analysis

    def select_option_trade(self, ticker, signal, underlying_price):
        """
        Based on the signal ("Bullish" or "Bearish"), fetch the options chain for the near-term expiry,
        and select an option contract meeting these conditions:
          - 1 day-to-expiry option.
          - Option premium (lastPrice) is below $3.00.
          - For bullish signals, select a call with strike > underlying price.
          - For bearish signals, select a put with strike < underlying price.
        Returns a dictionary with details of the selected option or None if no eligible option is found.
        """
        expiry = self.select_near_term_expiry(ticker)
        if not expiry:
            return None

        option_type = "call" if signal == "Bullish" else "put"
        chain_df = self.fetch_options_chain(ticker, expiry, option_type)
        if chain_df.empty:
            print(f"No {option_type} options data for {ticker} on expiry {expiry}.")
            return None

        # Filter contracts with a premium below $3.00
        eligible = chain_df[chain_df["lastPrice"] < 3.00].copy()
        if eligible.empty:
            print(f"No eligible {option_type} options under $3.00 for {ticker} on expiry {expiry}.")
            return None

        if signal == "Bullish":
            eligible = eligible[eligible["strike"] > underlying_price]
        else:
            eligible = eligible[eligible["strike"] < underlying_price]

        if eligible.empty:
            print(f"No {option_type} options meeting strike criteria for {ticker}.")
            return None

        # Choose the option with the strike closest to the underlying price
        eligible["diff"] = (eligible["strike"] - underlying_price).abs()
        selected = eligible.sort_values("diff").iloc[0]

        option_trade = {
            "ticker": ticker,
            "option_type": option_type.title(),
            "expiry": expiry,
            "strike": selected["strike"],
            "entry_option_price": selected["lastPrice"],
            "signal": signal
        }
        return option_trade

    def simulate_trade(self, ticker, option_trade, trade_timestamp, simulation_end):
        """
        Simulate the trade from the time of execution until simulation_end.
        Uses intraday underlying data (from trade_timestamp up to simulation_end) and a simplified delta (0.5)
        to estimate option price movement.
        Checks for profit target (20% gain) and stop loss (50% loss) triggers.
        Returns a dictionary with simulation details including exit time and simulated P&L (per contract, 100 shares).
        """
        profit_target = option_trade["entry_option_price"] * self.profit_target_multiplier
        stop_loss = option_trade["entry_option_price"] * self.stop_loss_multiplier
        delta = self.assumed_delta

        # Get the intraday data for the underlying starting at the trade_timestamp
        data = self.fetch_intraday_data(ticker)
        if data.empty:
            return None

        # Ensure trade_timestamp is timezone-aware (ET)
        et = pytz.timezone("US/Eastern")
        trade_time = trade_timestamp if trade_timestamp.tzinfo else et.localize(trade_timestamp)
        
        # Use simulation_end as provided (should be 10:30 ET for the trade at 10:00 ET)
        exit_time_limit = simulation_end

        # Filter data for simulation: after trade_time and at or before exit_time_limit
        sim_data = data[(data.index > trade_time) & (data.index <= exit_time_limit)]
        if sim_data.empty:
            print(f"No simulation data available for {ticker} after the trade time until {exit_time_limit}.")
            return None

        # Record the entry underlying price (using the price at trade_timestamp if available)
        if trade_time in data.index:
            entry_underlying_price = data.loc[trade_time, "Close"]
        else:
            entry_underlying_price = sim_data.iloc[0]["Close"]

        exit_price = None
        exit_timestamp = None
        trigger = None

        # Loop through simulation data to check if the profit target or stop loss is hit
        for idx, row in sim_data.iterrows():
            underlying_price = row["Close"]
            if option_trade["signal"] == "Bullish":
                simulated_option_price = option_trade["entry_option_price"] + delta * (underlying_price - entry_underlying_price)
            else:
                simulated_option_price = option_trade["entry_option_price"] + delta * (entry_underlying_price - underlying_price)

            if simulated_option_price >= profit_target:
                exit_price = profit_target
                exit_timestamp = idx
                trigger = "Profit Target"
                break
            elif simulated_option_price <= stop_loss:
                exit_price = stop_loss
                exit_timestamp = idx
                trigger = "Stop Loss"
                break

        if exit_price is None:
            # If neither level is triggered, exit at the last available simulation bar
            last_row = sim_data.iloc[-1]
            underlying_price = last_row["Close"]
            if option_trade["signal"] == "Bullish":
                exit_price = option_trade["entry_option_price"] + delta * (underlying_price - entry_underlying_price)
            else:
                exit_price = option_trade["entry_option_price"] + delta * (entry_underlying_price - underlying_price)
            exit_timestamp = sim_data.index[-1]
            trigger = "Time Exit"

        pnl = (exit_price - option_trade["entry_option_price"]) * 100  # one contract = 100 shares

        simulation = {
            "entry_option_price": option_trade["entry_option_price"],
            "exit_option_price": exit_price,
            "profit_target": profit_target,
            "stop_loss": stop_loss,
            "exit_timestamp": exit_timestamp,
            "trigger": trigger,
            "simulated_PnL": round(pnl, 2)
        }
        return simulation

    def run(self):
        """
        Run the bot continuously every minute from 9:30 a.m. ET until 10:30 a.m. ET.
        For each minute:
          - Before 10:00 a.m., the bot can perform routine checks.
          - At exactly 10:00 a.m. ET (if not already executed), the bot analyzes each ticker,
            selects a trade if a valid signal exists, and then simulates the trade until 10:30 a.m.
          - After 10:00 a.m., the bot continues to run until 10:30 a.m.
        """
        et = pytz.timezone("US/Eastern")
        today = datetime.datetime.now(et).date()
        start_time = et.localize(datetime.datetime.combine(today, datetime.time(9, 30)))
        end_time = et.localize(datetime.datetime.combine(today, datetime.time(10, 30)))
        
        print(f"Bot starting at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} and ending at {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}.")
        
        while True:
            now = datetime.datetime.now(et)
            print(f"\nCurrent ET time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Check if we're within our operating window
            if now < start_time:
                print("Waiting for 9:30 a.m. ET start...")
            elif now > end_time:
                print("Ending operating window. Exiting the bot loop.")
                break
            else:
                # At exactly 10:00 a.m. (using minute resolution) and not already executed:
                if now.strftime("%H:%M") == "10:00" and not self.trade_executed:
                    print("10:00 a.m. signal execution and trade tracking begins...")
                    # Set simulation_end to fixed 10:30 a.m. ET for trades executed at 10:00 a.m.
                    simulation_end = et.localize(datetime.datetime.combine(today, datetime.time(10, 30)))
                    
                    for ticker in self.tickers:
                        analysis = self.analyze_ticker(ticker)
                        if not analysis:
                            print(f"Insufficient data for {ticker}.")
                            continue

                        signal = analysis.get("signal")
                        if not signal:
                            print(f"No clear signal for {ticker} based on current indicators.")
                            continue

                        print(f"\nAnalysis for {ticker}: {analysis}")

                        option_trade = self.select_option_trade(ticker, signal, analysis["current_price"])
                        if option_trade is None:
                            print(f"No eligible option trade found for {ticker}.")
                            continue

                        trade_timestamp = analysis["timestamp"]
                        simulation = self.simulate_trade(ticker, option_trade, trade_timestamp, simulation_end)
                        if simulation is None:
                            print(f"Could not simulate trade for {ticker}.")
                            continue

                        suggestion = {
                            "ticker": ticker,
                            "signal": signal,
                            "underlying_price": analysis["current_price"],
                            "option_trade": option_trade,
                            "simulation": simulation
                        }
                        self.watchlist.append(suggestion)
                        print("-" * 60)
                        print(f"Ticker: {ticker}")
                        print(f"Signal: {signal}")
                        print(f"Underlying Price: {analysis['current_price']}")
                        print(f"Option Trade: {option_trade}")
                        print(f"Simulation Result: {simulation}")
                    
                    self.trade_executed = True
                else:
                    # For minutes that are not 10:00 or after the trade is executed,
                    # you can add other monitoring or logging if desired.
                    print("No trade execution at this minute.")
            
            # Sleep until the start of the next minute.
            seconds_to_next_minute = 60 - now.second
            time.sleep(seconds_to_next_minute)

# -------------------------------
# Running the Bot
# -------------------------------

if __name__ == "__main__":
    bot = OptionsTradingBot(config_file="/Users/etmadmin/Documents/Options Automation/config.json")
    bot.run()