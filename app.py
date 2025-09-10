import streamlit as st
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

# 프로젝트 내 다른 모듈 임포트
from config import (
    STOCK_UNIVERSE, DEFAULT_START_DATE, DEFAULT_END_DATE,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD, RSI_PERIOD,
    MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    INITIAL_CAPITAL, TRANSACTION_COST_PERCENT,
    MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE,
    TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_HOLD_DAYS,
    MAX_CONCURRENT_POSITIONS,
    PREDICTION_HORIZON_DAYS, TARGET_RISE_PERCENT,
    RSI_TREND_CONFIRM_LEVEL_BUY, RSI_TREND_CONFIRM_LEVEL_SELL,
    ML_MODEL_FEATURE_COLUMNS
)
from data_handler import get_prepared_data
from train_evaluate_ml import train_and_evaluate_models_for_streamlit
from ml_strategy import generate_signals_from_ml
from backtester import run_backtest
from performance_analyzer import calculate_performance_metrics, plot_performance
from strategy import generate_signals as generate_rule_signals

# 모델 저장 경로
MODEL_SAVE_DIR = './ml_model/'
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


# --- Streamlit 앱 기본 설정 ---
st.set_page_config(
    page_title="Quant Trading Backtesting System",
    page_icon="📈",
    layout="wide"
)

st.title("Automated Quant Trading Backtesting System 📈")
st.markdown("---")

# --- 1. 파라미터 설정 (사이드바) ---
st.sidebar.header("1. Strategy Parameters")

# 날짜 설정
st.sidebar.subheader("Date Range")
col1_date, col2_date = st.sidebar.columns(2)

with col1_date:
    start_date = st.date_input(
        "Start Date",
        value=datetime.datetime.strptime(DEFAULT_START_DATE, '%Y-%m-%d').date(),
        max_value=datetime.date.today()
    )

with col2_date:
    end_date = st.date_input(
    "End Date",
    value=datetime.datetime.strptime(DEFAULT_END_DATE, '%Y-%m-%d').date()
)


# 종목 유니버스 설정
st.sidebar.subheader("Stock Universe")
stock_universe_input = st.sidebar.text_input(
    "Enter Tickers (comma-separated)",
    value=", ".join(STOCK_UNIVERSE)
)
selected_stock_universe = [
    ticker.strip().upper() for ticker in stock_universe_input.split(',') if ticker.strip()
]

# 기술적 지표 파라미터
st.sidebar.subheader("Technical Indicators")
ema_short = st.sidebar.slider("EMA Short Period", 5, 50, EMA_SHORT_PERIOD)
ema_long = st.sidebar.slider("EMA Long Period", 30, 100, EMA_LONG_PERIOD)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, RSI_PERIOD)
macd_fast = st.sidebar.slider("MACD Fast Period", 5, 20, MACD_FAST_PERIOD)
macd_slow = st.sidebar.slider("MACD Slow Period", 20, 50, MACD_SLOW_PERIOD)
macd_signal = st.sidebar.slider("MACD Signal Period", 5, 15, MACD_SIGNAL_PERIOD)

# RSI 트렌드 레벨
rsi_buy = st.sidebar.slider("RSI Buy Level", 50, 70, RSI_TREND_CONFIRM_LEVEL_BUY)
rsi_sell = st.sidebar.slider("RSI Sell Level", 30, 50, RSI_TREND_CONFIRM_LEVEL_SELL)

# 백테스팅 파라미터
st.sidebar.subheader("Backtesting Rules")
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", 10000.0, 10000000.0, INITIAL_CAPITAL, step=10000.0
)
commission_rate = st.sidebar.slider(
    "Transaction Cost (%)", 0.0, 0.5, TRANSACTION_COST_PERCENT * 100, format="%.2f"
) / 100
max_alloc_per_trade = st.sidebar.slider(
    "Max Allocation per Trade (%)", 1, 100, int(MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE * 100)
) / 100
take_profit = st.sidebar.slider(
    "Take Profit (%)", 0.0, 50.0, TAKE_PROFIT_PERCENT * 100, format="%.1f"
) / 100
stop_loss = st.sidebar.slider(
    "Stop Loss (%)", 0.0, 50.0, STOP_LOSS_PERCENT * 100, format="%.1f"
) / 100
max_hold = st.sidebar.number_input("Max Hold Days", 10, 2000, MAX_HOLD_DAYS)
max_concurrent = st.sidebar.slider("Max Concurrent Positions", 1, 10, MAX_CONCURRENT_POSITIONS)

# ML 전략 관련 파라미터
st.sidebar.subheader("ML Strategy Specifics")
ml_horizon_days = st.sidebar.slider("ML Prediction Horizon Days", 5, 30, PREDICTION_HORIZON_DAYS)
ml_target_rise_pct = st.sidebar.slider(
    "ML Target Rise (%)", 1.0, 20.0, TARGET_RISE_PERCENT * 100, format="%.1f"
) / 100


# --- 2. 파이프라인 실행 버튼 ---
st.header("2. Run Pipeline Steps")

@st.cache_data(show_spinner="Preparing data...")
def cached_get_prepared_data(
    tickers, start_date_str, end_date_str,
    ema_short_period, ema_long_period, rsi_period,
    macd_fast_period, macd_slow_period, macd_signal_period,
    prediction_horizon_days, target_rise_percent
):
    st.info(f"Collecting and preparing data for: {', '.join(tickers)} from {start_date_str} to {end_date_str}")
    prepared_data = get_prepared_data(
        tickers=tickers,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        ema_short_period=ema_short_period,
        ema_long_period=ema_long_period,
        rsi_period=rsi_period,
        macd_fast_period=macd_fast_period,
        macd_slow_period=macd_slow_period,
        macd_signal_period=macd_signal_period,
        prediction_horizon_days=prediction_horizon_days,
        target_rise_percent=target_rise_percent
    )
    return prepared_data

# --- 데이터 준비 버튼 ---
if st.button("1. Prepare Data"):
    if not selected_stock_universe:
        st.error("Please enter at least one ticker in Stock Universe.")
    else:
        prepared_data_result = cached_get_prepared_data(
            selected_stock_universe,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            ema_short, ema_long, rsi_period,
            macd_fast, macd_slow, macd_signal,
            ml_horizon_days,
            ml_target_rise_pct
        )
        st.session_state.prepared_data = prepared_data_result

        if st.session_state.prepared_data is not None and not st.session_state.prepared_data.empty:
            st.success("Data prepared successfully!")
            st.write(f"Data period: {st.session_state.prepared_data.index.min().strftime('%Y-%m-%d')} ~ {st.session_state.prepared_data.index.max().strftime('%Y-%m-%d')}")
            st.write(f"Included Tickers: {st.session_state.prepared_data['Ticker'].unique().tolist()}")
            st.subheader("Raw Data Sample (First 5 Rows)")
            st.dataframe(st.session_state.prepared_data.head())
        else:
            st.error("Failed to prepare data. Please check parameters and network connection.")
 # --- ML 모델 학습 및 저장 ---
 st.header("3. ML Model Training")
 st.markdown("---")

# ML 학습 기간 UI
 col_ml_train_date1, col_ml_train_date2 = st.columns(2)
 with col_ml_train_date1:
     ml_train_start_date = st.date_input("ML Model Training Start Date",
                                          value=datetime.datetime.strptime(START_DATE, '%Y-%m-%d').date(),
                                          max_value=end_date - datetime.timedelta(days=PREDICTION_HORIZON_DAYS * 2))
 with col_ml_train_date2:
     default_end_date = end_date - datetime.timedelta(days=PREDICTION_HORIZON_DAYS)
     min_allowed_date = ml_train_start_date + datetime.timedelta(days=PREDICTION_HORIZON_DAYS * 2)
     if default_end_date < min_allowed_date:
         default_end_date = min_allowed_date

     ml_train_end_date = st.date_input("ML Model Training End Date",
                                        value=default_end_date,
                                        min_value=min_allowed_date)
