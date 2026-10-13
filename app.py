"""Daily-bar research UI; no broker orders."""
import datetime as dt
import pandas as pd
import streamlit as st
import config
from data_handler import get_prepared_data
from train_evaluate_ml import train_and_evaluate_models_for_streamlit
from ml_strategy import generate_signals_from_ml
from strategy import generate_signals
from backtester import run_backtest
from performance_analyzer import calculate_performance_metrics, plot_performance

st.set_page_config(page_title='Daily-bar strategy research',layout='wide')
st.title('Daily-bar strategy research')
st.caption('Close signals execute at later opens. Research simulation only.')
with st.sidebar:
    names=st.text_input('Tickers',', '.join(config.STOCK_UNIVERSE))
    start=st.date_input('Start date',dt.date.fromisoformat(config.START_DATE))
    end=st.date_input('End date',dt.date.fromisoformat(config.END_DATE),max_value=dt.date.today())
    short=st.slider('EMA short',5,50,config.EMA_SHORT_PERIOD)
    long=st.slider('EMA long',30,100,config.EMA_LONG_PERIOD)
    rsi=st.slider('RSI period',5,30,config.RSI_PERIOD)
    horizon=st.slider('Prediction horizon (sessions)',1,30,config.PREDICTION_HORIZON_DAYS)
    rise=st.slider('Target rise (%)',1.,20.,config.TARGET_RISE_PERCENT*100)/100
    capital=st.number_input('Initial capital',1000.,10000000.,config.INITIAL_CAPITAL)
    fee=st.slider('Per-side commission (%)',0.,.5,config.TRANSACTION_COST_PERCENT*100)/100
    allocation=st.slider('Maximum allocation per entry (%)',1,100,int(config.MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE*100))/100
    tp=st.slider('Close-observed take profit (%)',0.,50.,config.TAKE_PROFIT_PERCENT*100)/100
    sl=st.slider('Close-observed stop loss (%)',0.,50.,config.STOP_LOSS_PERCENT*100)/100
    hold=st.number_input('Maximum held sessions',1,2000,config.MAX_HOLD_DAYS)
    positions=st.slider('Maximum positions',1,20,config.MAX_CONCURRENT_POSITIONS)
    buy=st.slider('RSI buy threshold',0,100,config.RSI_TREND_CONFIRM_LEVEL_BUY)
    sell=st.slider('RSI sell threshold',0,100,config.RSI_TREND_CONFIRM_LEVEL_SELL)
tickers=[t.strip().upper() for t in names.split(',') if t.strip()]
signature=(tuple(tickers),str(start),str(end),short,long,rsi,horizon,rise)
if st.session_state.get('prepared_signature') not in (None,signature):
    for key in ['prepared_data','prepared_signature','ml_ready','portfolio_df','trades_df']:
        st.session_state.pop(key,None)
    st.info('Settings changed. Prepare data and retrain.')
if st.button('1. Prepare data'):
    try:
        with st.spinner('Downloading adjusted daily bars'):
            frame=get_prepared_data(tickers,str(start),str(end),short,long,rsi,config.MACD_FAST_PERIOD,config.MACD_SLOW_PERIOD,config.MACD_SIGNAL_PERIOD,horizon,rise)
        st.session_state.prepared_data=frame
        st.session_state.prepared_signature=signature
        st.session_state.pop('ml_ready',None)
        st.success(f'Prepared {len(frame)} rows; unknown targets remain missing.')
    except Exception as exc: st.error(str(exc))
frame=st.session_state.get('prepared_data')
fraction=st.slider('History reserved for training/validation (%)',40,80,70)/100
if st.button('2. Train ML models'):
    try:
        if frame is None: raise ValueError('Prepare data first')
        dates=frame.index.unique().sort_values()
        boundary=dates[int(len(dates)*fraction)]
        with st.spinner('Training chronological purged data'):
            metrics,_=train_and_evaluate_models_for_streamlit(frame.loc[frame.index<boundary].copy(),tickers,horizon,rise)
        st.session_state.ml_ready=True
        st.json(metrics)
    except Exception as exc:
        st.session_state.pop('ml_ready',None)
        st.error(str(exc))
strategy=st.radio('Strategy',['Rules-Based','ML-Based'])
if st.button('3. Run backtest'):
    try:
        if frame is None: raise ValueError('Prepare data first')
        if strategy=='ML-Based':
            if not st.session_state.get('ml_ready'): raise ValueError('Train current data first')
            signals=generate_signals_from_ml(frame,'ml_model')
        else: signals=generate_signals(frame,rsi_buy_level=buy,rsi_sell_level=sell)
        merged=frame.reset_index().merge(signals.reset_index(),on=['Date','Ticker'],validate='one_to_one').set_index('Date')
        portfolio,trades=run_backtest(merged,capital,fee,allocation,tp,sl,int(hold),positions)
        st.json(calculate_performance_metrics(portfolio,capital))
        st.pyplot(plot_performance(portfolio,merged[merged.Ticker==tickers[0]],tickers[0]))
        st.dataframe(trades)
    except Exception as exc: st.error(str(exc))
if frame is not None: st.dataframe(frame.tail(10))
