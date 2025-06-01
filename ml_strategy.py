# ml_strategy.py (Streamlit 연동을 위해 ML_MODEL_FEATURE_COLUMNS를 파라미터로 받도록 수정)

import pandas as pd
import joblib
import os
import streamlit as st # Streamlit Cloud 배포를 위해 st 임포트 (로그용)

# ML_MODEL_FEATURE_COLUMNS는 이제 함수 파라미터로 전달받게 됩니다.
# 따라서 이 전역 변수는 필요 없으므로 주석 처리하거나 제거합니다.
# ML_MODEL_FEATURE_COLUMNS = [...] 

# ML 모델이 예상하는 피처 컬럼 목록을 정의합니다.
ML_MODEL_FEATURE_COLUMNS = [
    'open',
    'high',
    'low',
    'close',
    'volume',
    'RSI',
    'MACD',
    'BollingerBands_Upper',
    'BollingerBands_Lower',
    # ML 모델이 사용하는 다른 모든 피처를 추가하세요
]

def generate_signals_from_ml(df_with_indicators, model_dir, ml_model_feature_columns):
    """
    학습된 ML 모델을 사용하여 각 종목별 매매 신호를 생성합니다.
    신호: 1 (매수), 0 (보유/관망).

    Args:
        df_with_indicators (pd.DataFrame): 'Ticker' 컬럼 및 모델 학습에 사용된
                                          모든 피처 컬럼을 포함하는 DataFrame.
                                          data_handler.get_prepared_data()의 반환값.
        model_dir (str): 학습된 .pkl 모델 파일들이 저장된 디렉토리 경로.
        ml_model_feature_columns (list): 모델 학습에 사용된 최종 피처 컬럼 리스트.

    Returns:
        pd.DataFrame: 'Date'를 인덱스로 하고 'signal'과 'Ticker' 컬럼을 가진 DataFrame.
    """
    all_signals = []
    st.info(f"ML signal generation started. Using model directory: {model_dir}")
    st.write(f"Number of features for prediction: {len(ml_model_feature_columns)}")
    st.write(f"Example features (first 5): {ml_model_feature_columns[:5]}")

    # df_with_indicators에 있는 모든 티커에 대해 반복
    for ticker in df_with_indicators['Ticker'].unique():
        ticker_df = df_with_indicators[df_with_indicators['Ticker'] == ticker].copy()

        if ticker_df.empty:
            st.info(f"Info: No data for {ticker}. Skipping signal generation.")
            continue

        # 예측에 사용할 피처만 선택 (ML_MODEL_FEATURE_COLUMNS 사용)
        missing_features = [col for col in ml_model_feature_columns if col not in ticker_df.columns]
        if missing_features:
            st.warning(f"Warning: Missing required features for {ticker}: {missing_features}. Skipping this ticker.")
            signals_for_ticker_empty = pd.DataFrame(index=ticker_df.index)
            signals_for_ticker_empty['signal'] = 0.0
            signals_for_ticker_empty['Ticker'] = ticker
            all_signals.append(signals_for_ticker_empty[['signal', 'Ticker']])
            continue

        X_live = ticker_df[ml_model_feature_columns]

        if X_live.isnull().values.any():
            nan_count = X_live.isnull().sum().sum()
            st.warning(f"Warning: {ticker} prediction input data has {nan_count} NaN values. Filling with 0.")
            X_live = X_live.fillna(0)


        if X_live.empty:
            st.warning(f"Warning: No valid feature data for prediction for {ticker}. Skipping.")
            signals_for_ticker_empty = pd.DataFrame(index=ticker_df.index)
            signals_for_ticker_empty['signal'] = 0.0
            signals_for_ticker_empty['Ticker'] = ticker
            all_signals.append(signals_for_ticker_empty[['signal', 'Ticker']])
            continue

        signals_for_ticker = pd.DataFrame(index=ticker_df.index)
        signals_for_ticker['signal'] = 0.0
        signals_for_ticker['Ticker'] = ticker

        model_filename = f'ensemble_voting_classifier_hard_{ticker}.pkl'
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            st.warning(f"Warning: Model file for {ticker} ({model_path}) not found. Using 0 signal.")
            all_signals.append(signals_for_ticker[['signal', 'Ticker']])
            continue

        try:
            loaded_model = joblib.load(model_path)
            predictions = loaded_model.predict(X_live)
            signals_for_ticker['signal'] = predictions.astype(float)
        except ValueError as ve:
            st.error(f"Error: ValueError during model prediction for {ticker} (feature mismatch, etc.): {ve}")
        except Exception as e:
            st.error(f"Error: Other error during model load or prediction for {ticker}: {e}")

        all_signals.append(signals_for_ticker[['signal', 'Ticker']])

    if not all_signals:
        st.warning("Warning: Failed to generate ML signals for all tickers.")
        empty_df = pd.DataFrame(columns=['signal', 'Ticker'])
        empty_df.index.name = 'Date'
        return empty_df

    final_signals_df = pd.concat(all_signals).sort_index()
    st.write(f"ML signal generation completed. Total signals: {len(final_signals_df)}")
    return final_signals_df
