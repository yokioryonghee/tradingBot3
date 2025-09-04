# app.py (수정된 전체 코드 - 최종본)

import streamlit as st
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

# 프로젝트 내 다른 모듈 임포트
from config import (
    STOCK_UNIVERSE, START_DATE, END_DATE,
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD, RSI_PERIOD,
    MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    INITIAL_CAPITAL, TRANSACTION_COST_PERCENT,
    MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE,
    TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, MAX_HOLD_DAYS,
    MAX_CONCURRENT_POSITIONS,
    PREDICTION_HORIZON_DAYS, TARGET_RISE_PERCENT,
    RSI_TREND_CONFIRM_LEVEL_BUY, RSI_TREND_CONFIRM_LEVEL_SELL,
    ML_MODEL_FEATURE_COLUMNS # <-- config에서 ML_MODEL_FEATURE_COLUMNS를 임포트합니다.
)
from data_handler import get_prepared_data
from train_evaluate_ml import train_and_evaluate_models_for_streamlit
from ml_strategy import generate_signals_from_ml # ML_MODEL_FEATURE_COLUMNS는 여기서 임포트하지 않습니다.
from backtester import run_backtest
from performance_analyzer import calculate_performance_metrics, plot_performance

# 순환 임포트 문제를 해결했으므로 다시 활성화합니다.
from strategy import generate_signals as generate_rule_signals # <--- 이 줄을 활성화해야 합니다.

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
    start_date = st.date_input("Start Date",
                                value=datetime.datetime.strptime(START_DATE, '%Y-%m-%d').date(),
                                max_value=datetime.date.today() - datetime.timedelta(days=365*5))
with col2_date:
    end_date = st.date_input("End Date",
                              value=datetime.datetime.strptime(END_DATE, '%Y-%m-%d').date(),
                              max_value=datetime.date.today())

# 종목 유니버스 설정
st.sidebar.subheader("Stock Universe")
stock_universe_input = st.sidebar.text_input("Enter Tickers (comma-separated)",
                                            value=", ".join(STOCK_UNIVERSE))
selected_stock_universe = [ticker.strip().upper() for ticker in stock_universe_input.split(',') if ticker.strip()]

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
initial_capital = st.sidebar.number_input("Initial Capital ($)", 10000.0, 10000000.0, INITIAL_CAPITAL, step=10000.0)
commission_rate = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, TRANSACTION_COST_PERCENT * 100, format="%.2f") / 100
max_alloc_per_trade = st.sidebar.slider("Max Allocation per Trade (%)", 1, 100, int(MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE * 100)) / 100
take_profit = st.sidebar.slider("Take Profit (%)", 0.0, 50.0, TAKE_PROFIT_PERCENT * 100, format="%.1f") / 100
stop_loss = st.sidebar.slider("Stop Loss (%)", 0.0, 50.0, STOP_LOSS_PERCENT * 100, format="%.1f") / 100
max_hold = st.sidebar.number_input("Max Hold Days", 10, 2000, MAX_HOLD_DAYS)
max_concurrent = st.sidebar.slider("Max Concurrent Positions", 1, 10, MAX_CONCURRENT_POSITIONS)

# ML 전략 관련 파라미터
st.sidebar.subheader("ML Strategy Specifics")
ml_horizon_days = st.sidebar.slider("ML Prediction Horizon Days", 5, 30, PREDICTION_HORIZON_DAYS)
ml_target_rise_pct = st.sidebar.slider("ML Target Rise (%)", 1.0, 20.0, TARGET_RISE_PERCENT * 100, format="%.1f") / 100


# --- 2. 파이프라인 실행 버튼 ---
st.header("2. Run Pipeline Steps")

@st.cache_data(show_spinner="Preparing data...")
def cached_get_prepared_data(tickers, start_date_str, end_date_str,
                             ema_short_period, ema_long_period, rsi_period,
                             macd_fast_period, macd_slow_period, macd_signal_period,
                             prediction_horizon_days, target_rise_percent):
    st.info(f"Collecting and preparing data for: {', '.join(tickers)} from {start_date_str} to {end_date_str}")
    prepared_data = get_prepared_data(tickers=tickers,
                                       start_date_str=start_date_str,
                                       end_date_str=end_date_str,
                                       ema_short_period=ema_short_period,
                                       ema_long_period=ema_long_period,
                                       rsi_period=rsi_period,
                                       macd_fast_period=macd_fast_period,
                                       macd_slow_period=macd_slow_period,
                                       macd_signal_period=macd_signal_period,
                                       prediction_horizon_days=prediction_horizon_days,
                                       target_rise_percent=target_rise_percent)
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
    ml_train_end_date = st.date_input("ML Model Training End Date",
                                       value=end_date - datetime.timedelta(days=PREDICTION_HORIZON_DAYS),
                                       min_value=ml_train_start_date + datetime.timedelta(days=PREDICTION_HORIZON_DAYS * 2))

if st.button("2. Train ML Models"):
    if "prepared_data" not in st.session_state or st.session_state.prepared_data is None or st.session_state.prepared_data.empty:
        st.error("Please prepare data first (Step 1).")
    else:
        train_data_for_ml = st.session_state.prepared_data.loc[
            (st.session_state.prepared_data.index >= pd.Timestamp(ml_train_start_date)) &
            (st.session_state.prepared_data.index <= pd.Timestamp(ml_train_end_date))
        ].copy()

        if train_data_for_ml.empty:
            st.warning("No data available for the selected ML training period. Adjust dates in sidebar or Training Date Range.")
        else:
            with st.spinner("Training ML models... This might take a while."):
                model_metrics, feature_columns_trained = train_and_evaluate_models_for_streamlit(
                    train_data_for_ml,
                    tickers=selected_stock_universe,
                    prediction_horizon_days=ml_horizon_days,
                    target_rise_percent=ml_target_rise_pct
                )

                if model_metrics:
                    st.session_state.model_metrics = model_metrics
                    st.session_state.ml_model_trained = True
                    st.session_state.ml_feature_columns = feature_columns_trained # 학습에 사용된 피처 컬럼 저장

                    st.success("ML models trained and evaluated successfully!")
                    st.subheader("ML Model Evaluation Metrics")
                    for ticker, metrics in model_metrics.items():
                        st.markdown(f"**{ticker} Metrics:**")
                        st.json(metrics)

                else:
                    st.error("ML model training failed. Check your data and parameters.")
                    st.session_state.ml_model_trained = False

# --- 4. 백테스팅 실행 ---
st.header("4. Run Backtest")
st.markdown("---")

strategy_type = st.radio("Select Strategy Type:", ("Rules-Based", "ML-Based"), key="strategy_selector")

if strategy_type == "ML-Based" and ("ml_model_trained" not in st.session_state or not st.session_state.ml_model_trained):
    st.warning("Please train ML models first (Step 2) to use ML-Based strategy.")

# ML 백테스팅 기간 UI 추가
if strategy_type == "ML-Based":
    st.subheader("ML Backtesting Period")
    col_ml_bt_date1, col_ml_bt_date2 = st.columns(2)
    with col_ml_bt_date1:
        # ML 학습 종료일 다음 날부터 시작하도록 기본값 설정
        ml_backtest_start_date = st.date_input("ML Backtesting Start Date",
                                            value=ml_train_end_date + datetime.timedelta(days=1),
                                            min_value=ml_train_end_date + datetime.timedelta(days=1), # 최소값 제한
                                            max_value=end_date)
    with col_ml_bt_date2:
        ml_backtest_end_date = st.date_input("ML Backtesting End Date",
                                          value=end_date, # 전체 데이터 종료일 기본값
                                          min_value=ml_backtest_start_date) # 시작일보다 작을 수 없음
elif strategy_type == "Rules-Based":
    st.subheader("Rules-Based Backtesting Period")
    col_rb_bt_date1, col_rb_bt_date2 = st.columns(2)
    with col_rb_bt_date1:
        rule_backtest_start_date = st.date_input("Rules-Based Backtesting Start Date",
                                                  value=start_date, # 데이터 로드 시작일부터 시작
                                                  max_value=end_date)
    with col_rb_bt_date2:
        rule_backtest_end_date = st.date_input("Rules-Based Backtesting End Date",
                                                value=end_date,
                                                min_value=rule_backtest_start_date)

if st.button("3. Run Backtest"):
    if "prepared_data" not in st.session_state or st.session_state.prepared_data is None or st.session_state.prepared_data.empty:
        st.error("Please prepare data first (Step 1).")
    elif strategy_type == "ML-Based" and ("ml_model_trained" not in st.session_state or not st.session_state.ml_model_trained):
        st.error("Please train ML models first (Step 2) to run ML-Based strategy.")
    else:
        with st.spinner("Running backtest..."):

            bt_params = {
                "initial_capital": initial_capital,
                "commission_rate": commission_rate,
                "max_allocation_pct_per_trade": max_alloc_per_trade,
                "take_profit_pct": take_profit,
                "stop_loss_pct": stop_loss,
                "max_hold_days": max_hold,
                "max_concurrent_positions": max_concurrent
            }

            signals_df = pd.DataFrame() # 초기화
            base_backtest_data_for_merge = pd.DataFrame() # 초기화

            if strategy_type == "Rules-Based":
                # 규칙 기반 전략은 선택된 백테스팅 기간의 데이터를 사용
                backtest_data_for_rules = st.session_state.prepared_data.loc[
                    (st.session_state.prepared_data.index >= pd.Timestamp(rule_backtest_start_date)) &
                    (st.session_state.prepared_data.index <= pd.Timestamp(rule_backtest_end_date))
                ].copy()

                if backtest_data_for_rules.empty:
                    st.error("No data available for the selected Rules-Based backtesting period. Adjust dates.")
                else:
                    st.info(f"Rules-Based Backtest Data Period: {backtest_data_for_rules.index.min().strftime('%Y-%m-%d')} ~ {backtest_data_for_rules.index.max().strftime('%Y-%m-%d')}")
                    signals_df = generate_rule_signals(
                        backtest_data_for_rules, # 필터링된 데이터를 전달
                        ema_short_period=ema_short,
                        ema_long_period=ema_long,
                        rsi_period=rsi_period,
                        rsi_buy_level=rsi_buy,
                        rsi_sell_level=rsi_sell
                    )
                    base_backtest_data_for_merge = backtest_data_for_rules # 병합을 위해 사용

            elif strategy_type == "ML-Based":
                MODEL_DIR = "./ml_model/"
                if "ml_feature_columns" not in st.session_state:
                    st.error("ML feature columns not found in session state. Please train models again.")
                else:
                    # ML 백테스팅 기간에 맞게 데이터 필터링
                    backtest_data_for_ml = st.session_state.prepared_data.loc[
                        (st.session_state.prepared_data.index >= pd.Timestamp(ml_backtest_start_date)) &
                        (st.session_state.prepared_data.index <= pd.Timestamp(ml_backtest_end_date))
                    ].copy()

                    if backtest_data_for_ml.empty:
                        st.error("No data available for the selected ML backtesting period. Adjust dates.")
                    else:
                        st.info(f"ML Backtest Data Period: {backtest_data_for_ml.index.min().strftime('%Y-%m-%d')} ~ {backtest_data_for_ml.index.max().strftime('%Y-%m-%d')}")
                        signals_df = generate_signals_from_ml(
                            backtest_data_for_ml, # 필터링된 데이터를 전달
                            MODEL_DIR,
                            st.session_state.ml_feature_columns # 학습된 피처 컬럼 전달
                        )
                        base_backtest_data_for_merge = backtest_data_for_ml # 병합을 위해 사용

            if signals_df.empty:
                st.error(f"{strategy_type} signal generation failed. No signals were generated.")
            else:
                st.info(f"Generated {len(signals_df)} signals for {strategy_type} strategy.")

                # 백테스팅을 위한 데이터 통합
                data_to_merge = base_backtest_data_for_merge.reset_index() # 각 전략 유형에 맞는 데이터 사용
                signals_to_merge = signals_df.reset_index()

                data_for_backtest_full = pd.merge(
                    data_to_merge,
                    signals_to_merge[['Date', 'Ticker', 'signal']],
                    on=['Date', 'Ticker'],
                    how='inner'
                )
                data_for_backtest_full.set_index('Date', inplace=True)
                data_for_backtest_full.sort_index(inplace=True)

                required_cols = ['CLOSE', 'VOLUME', 'volume_change_pct', 'Ticker', 'signal']
                missing_cols = [col for col in required_cols if col not in data_for_backtest_full.columns]
                if missing_cols:
                    st.error(f"Error: Missing required columns for backtest: {missing_cols}")
                else:
                    data_for_backtest_final = data_for_backtest_full[required_cols]

                    portfolio_df, trades_df = run_backtest(data_for_backtest_final, **bt_params)

                    if not portfolio_df.empty:
                        st.success("Backtest completed successfully!")
                        st.session_state.portfolio_df = portfolio_df
                        st.session_state.trades_df = trades_df

                        # 성과 분석 및 시각화
                        st.subheader("Performance Analysis")
                        performance_metrics = calculate_performance_metrics(portfolio_df, initial_capital)
                        buy_trades = len(trades_df[trades_df['Type'] == 'BUY']) if not trades_df.empty else 0
                        sell_trades = len(trades_df[trades_df['Type'] == 'SELL']) if not trades_df.empty else 0
                        performance_metrics["Number of Buy Trades"] = buy_trades
                        performance_metrics["Number of Sell Trades"] = sell_trades

                        st.json(performance_metrics)

                        if not trades_df.empty:
                            st.subheader("Recent Trades (Last 5)")
                            st.dataframe(trades_df.tail())

                        st.subheader("Portfolio Value Over Time")

                        display_ticker = selected_stock_universe[0] if selected_stock_universe else "N/A"
                        # 벤치마크 데이터도 백테스팅 기간에 맞춰 필터링
                        benchmark_data_for_plot = st.session_state.prepared_data[st.session_state.prepared_data['Ticker'] == display_ticker].copy()

                        if not portfolio_df.empty:
                            plot_start_date = portfolio_df.index.min()
                            plot_end_date = portfolio_df.index.max() # 백테스팅 종료 날짜
                            benchmark_data_for_plot = benchmark_data_for_plot.loc[
                                (benchmark_data_for_plot.index >= plot_start_date) &
                                (benchmark_data_for_plot.index <= plot_end_date)
                            ].copy()

                        fig = plot_performance(portfolio_df, benchmark_data_for_plot, display_ticker)
                        st.pyplot(fig)
                    else:
                        st.error("Backtest resulted in an empty portfolio.")

# --- 앱 초기화 버튼 ---
if st.sidebar.button("Reset All Data & Parameters"):
    st.session_state.clear()
    st.rerun()
