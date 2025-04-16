import json
import datetime
import pytz
import time
import logging
import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------------
# Configure Logging
# -------------------------------

# Create file names using the current date
et = pytz.timezone("US/Eastern")
now = datetime.datetime.now(et)
date_str = now.strftime("%m%d%Y")
general_log_filename = f"logs_{date_str}.txt"
trade_log_filename = f"logs_trades_{date_str}.txt"

# Configure a logger for general output
general_logger = logging.getLogger("general")
general_logger.setLevel(logging.INFO)
gen_handler = logging.FileHandler(general_log_filename)
gen_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
gen_handler.setFormatter(gen_formatter)
general_logger.addHandler(gen_handler)

# Also output general logs to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(gen_formatter)
general_logger.addHandler(console_handler)

# Configure a separate logger for trade recommendations
trade_logger = logging.getLogger("trade")
trade_logger.setLevel(logging.INFO)
trade_handler = logging.FileHandler(trade_log_filename)
trade_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)

# -------------------------------
# Technical Indicator Functions
# -------------------------------

def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index (RSI) for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(series, period=20):
    """Calculate the simple moving average (SMA) for a price series."""
    return series.rolling(window=period).mean()

def calculate_bollinger_bands(series, period=20, num_std=2):
    """Calculate Bollinger Bands based on the SMA and standard deviation."""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

# -------------------------------
# Options Trading Bot Class
# -------------------------------

class OptionsTradingBot:
    def __init__(self, config_file='config.json', testing=True):
        # Load tickers from config file.
        self.config = self.load_config(config_file)
        self.tickers = self.config.get("tickers", [])
        self.watchlist = []  # To store trade suggestions and simulation details.
        
        # Trade simulation parameters.
        self.profit_target_multiplier = 1.20
        self.stop_loss_multiplier = 0.50
        self.assumed_delta = 0.5  # Simplified delta for simulation.
        
        # When testing is True, use shorter lookback periods.
        self.testing = testing

    def load_config(self, filename):
        """Load configuration from a JSON file."""
        try:
            with open(filename, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            general_logger.error(f"Error loading config file {filename}: {e}")
            return {}

    def fetch_intraday_data(self, ticker, period='1d', interval='5m'):
        """
        Fetch intraday data using yfinance.
        Returns a DataFrame.
        """
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                general_logger.info(f"No intraday data for {ticker}.")
            return data
        except Exception as e:
            general_logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return pd.DataFrame()

    def select_near_term_expiry(self, ticker):
        """
        Select an expiry date approximately 1 day-to-expiry.
        If an expiry exactly one day away is not available, choose the nearest future expiry.
        """
        ticker_obj = yf.Ticker(ticker)
        available_exps = ticker_obj.options
        if not available_exps:
            general_logger.info(f"No options expiration data for {ticker}.")
            return None

        now_date = datetime.datetime.now(pytz.timezone('US/Eastern')).date()
        candidate = None
        min_diff = None
        for exp in available_exps:
            try:
                exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            except Exception:
                continue
            diff = (exp_date - now_date).days
            if diff >= 1:
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    candidate = exp
        if candidate:
            return candidate
        else:
            general_logger.info(f"No suitable expiry found for {ticker}.")
            return None

    def fetch_options_chain(self, ticker, expiry, option_type="call"):
        """
        Fetch the options chain for a given ticker and expiry date.
        Returns a DataFrame for calls or puts.
        """
        ticker_obj = yf.Ticker(ticker)
        try:
            chain = ticker_obj.option_chain(expiry)
            if option_type.lower() == "call":
                return chain.calls
            elif option_type.lower() == "put":
                return chain.puts
        except Exception as e:
            general_logger.error(f"Error fetching options chain for {ticker}: {e}")
        return pd.DataFrame()

    def analyze_ticker(self, ticker):
        """
        Analyze the underlying ticker using intraday data.
        Generates a signal if:
          - Bullish (oversold): RSI < 30 and price near or below the lower Bollinger Band.
          - Bearish (overbought): RSI > 70 and price near or above the upper Bollinger Band.
        Returns a dictionary of analysis details or None.
        """
        data = self.fetch_intraday_data(ticker)
        if data.empty:
            return None
        
        # Set periods depending on testing mode.
        if self.testing:
            required_bars = 5   # Use fewer data points
            rsi_period = 5
            sma_period = 5
        else:
            required_bars = 20
            rsi_period = 14
            sma_period = 20

        if len(data) < required_bars:
            general_logger.info(f"Insufficient data for {ticker}: only {len(data)} bars available.")
            return None

        # Calculate indicators.
        data["RSI"] = calculate_rsi(data["Close"], period=rsi_period)
        sma, upper_band, lower_band = calculate_bollinger_bands(data["Close"], period=sma_period)
        data["SMA"] = sma
        data["UpperBand"] = upper_band
        data["LowerBand"] = lower_band

        latest = data.iloc[-1]
        # Use .item() to get Python scalars.
        try:
            close = latest["Close"].item()
            rsi_val = latest["RSI"].item()
            lower_band_val = latest["LowerBand"].item()
            upper_band_val = latest["UpperBand"].item()
        except Exception as e:
            general_logger.error(f"Error extracting scalar values for {ticker}: {e}")
            return None

        tolerance = 0.01  # 1% tolerance
        signal = None
        if (rsi_val < 40) or (close <= lower_band_val * (1 + 0.05)):
            signal = "Bullish"
        elif (rsi_val > 60) or (close >= upper_band_val * (1 - 0.05)):
            signal = "Bearish"

        analysis = {
            "ticker": ticker,
            "current_price": close,
            "RSI": round(rsi_val, 2),
            "SMA": round(latest["SMA"].item(), 2),
            "UpperBand": round(upper_band_val, 2),
            "LowerBand": round(lower_band_val, 2),
            "signal": signal,
            "timestamp": latest.name  # Time of the latest data point.
        }
        return analysis

    def select_option_trade(self, ticker, signal, underlying_price):
        """
        For a given ticker and signal, fetch the near-term options chain and select an eligible option.
        Conditions:
          - Approximately 1 day-to-expiry.
          - Option premium below $3.00.
          - Bullish: select OTM call (strike > underlying price).
          - Bearish: select OTM put (strike < underlying price).
        Returns a dictionary with option trade details or None.
        """
        expiry = self.select_near_term_expiry(ticker)
        if not expiry:
            return None

        option_type = "call" if signal == "Bullish" else "put"
        chain_df = self.fetch_options_chain(ticker, expiry, option_type)
        if chain_df.empty:
            general_logger.info(f"No {option_type} options data for {ticker} with expiry {expiry}.")
            return None

        eligible = chain_df[chain_df["lastPrice"] < 3.00].copy()
        if eligible.empty:
            general_logger.info(f"No eligible {option_type} options under $3.00 for {ticker} on {expiry}.")
            return None

        if signal == "Bullish":
            eligible = eligible[eligible["strike"] >= underlying_price]
        else:
            eligible = eligible[eligible["strike"] <= underlying_price]

        if eligible.empty:
            general_logger.info(f"No {option_type} options meeting strike criteria for {ticker}.")
            return None

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
        Simulate a trade from the given trade_timestamp until simulation_end.
        Uses a simplified delta (0.5) on underlying price to approximate option price movement.
        Checks for a 20% profit target or a 50% stop loss.
        Returns a dictionary with simulation details and calculated P&L.
        """
        profit_target = option_trade["entry_option_price"] * self.profit_target_multiplier
        stop_loss = option_trade["entry_option_price"] * self.stop_loss_multiplier
        delta = self.assumed_delta

        data = self.fetch_intraday_data(ticker)
        if data.empty:
            return None

        et = pytz.timezone("US/Eastern")
        trade_time = trade_timestamp if trade_timestamp.tzinfo else et.localize(trade_timestamp)
        sim_data = data[(data.index > trade_time) & (data.index <= simulation_end)]
        if sim_data.empty:
            general_logger.info(f"No simulation data available for {ticker} between {trade_time} and {simulation_end}.")
            return None

        if trade_time in data.index:
            entry_underlying_price = data.loc[trade_time, "Close"]
        else:
            entry_underlying_price = sim_data.iloc[0]["Close"]

        exit_price = None
        exit_timestamp = None
        trigger = None

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
        Run the bot continuously—without any operating time constraint.
        In each iteration, it:
          - Analyzes each ticker.
          - Logs the analysis output to the general log.
          - If a clear signal is detected for a ticker, selects an option trade,
            simulates the trade over a preset window, and logs the trade details to the trade log.
          - Waits for a minute before the next iteration.
        """
        et = pytz.timezone("US/Eastern")
        general_logger.info("Starting indefinite run. Press Ctrl+C to exit.")

        while True:
            now = datetime.datetime.now(et)
            # For each ticker in the config, perform analysis.
            for ticker in self.tickers:
                analysis = self.analyze_ticker(ticker)
                if analysis is None:
                    general_logger.info(f"Ticker {ticker}: Insufficient data or error during analysis.")
                    continue

                general_logger.info(f"Ticker {ticker} analysis: {analysis}")

                signal = analysis.get("signal")
                if not signal:
                    general_logger.info(f"Ticker {ticker}: No clear signal based on current indicators.")
                    continue

                # If there is a clear signal, attempt to select an option trade.
                option_trade = self.select_option_trade(ticker, signal, analysis["current_price"])
                if option_trade is None:
                    general_logger.info(f"Ticker {ticker}: No eligible option trade found.")
                    continue

                # For simulation, run for 30 minutes from now.
                simulation_end = now + datetime.timedelta(minutes=30)
                trade_timestamp = analysis["timestamp"]
                simulation = self.simulate_trade(ticker, option_trade, trade_timestamp, simulation_end)
                if simulation is None:
                    general_logger.info(f"Ticker {ticker}: Could not simulate the trade.")
                    continue

                suggestion = {
                    "ticker": ticker,
                    "signal": signal,
                    "underlying_price": analysis["current_price"],
                    "option_trade": option_trade,
                    "simulation": simulation
                }
                # Save suggestion in the watchlist.
                self.watchlist.append(suggestion)
                general_logger.info(f"Trade Recommendation for {ticker}: {suggestion}")
                trade_logger.info(f"Trade Recommendation for {ticker}: {suggestion}")
            
            # Sleep for 60 seconds before next iteration.
            time.sleep(60)

# -------------------------------
# Running the Bot Indefinitely
# -------------------------------

if __name__ == "__main__":
    bot = OptionsTradingBot(config_file="config.json", testing=True)
    try:
        bot.run()
    except KeyboardInterrupt:
        general_logger.info("Exiting due to keyboard interruption.")