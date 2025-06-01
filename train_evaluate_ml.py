# train_evaluate_ml.py (Streamlit 연동 버전 - 최종 수정)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib
import os
import streamlit as st # Streamlit Cloud 배포를 위해 st 임포트 (로그용)

# 모델 저장 경로 설정 (Streamlit Cloud에서도 접근 가능하도록)
MODEL_SAVE_PATH = './ml_model/' # 현재 디렉토리 기준 ml_model 폴더
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)


def train_and_evaluate_models_for_streamlit(
    prepared_df, # 이미 준비된 데이터프레임
    tickers,
    prediction_horizon_days, # config.py에서 받지 않고 파라미터로 받음
    target_rise_percent # config.py에서 받지 않고 파라미터로 받음
):
    """
    Streamlit 앱에서 호출될 ML 모델 학습 및 평가 함수.
    Args:
        prepared_df (pd.DataFrame): 'Ticker' 컬럼 및 모든 피처/타겟을 포함하는 DataFrame.
        tickers (list): 학습할 종목 티커 리스트.
        prediction_horizon_days (int): 타겟 변수 계산에 사용될 예측 기간.
        target_rise_percent (float): 타겟 변수 계산에 사용될 목표 상승률.

    Returns:
        tuple: (dict: 각 종목별 학습 및 평가 지표, list: 학습에 사용된 최종 피처 컬럼 리스트)
    """
    
    all_model_metrics = {}
    final_feature_columns_used = [] # 학습에 사용된 최종 피처 컬럼을 저장할 리스트

    target_column = 'target'
    excluded_cols_for_training = [target_column, 'Ticker', 'future_high', 'ADJ CLOSE']
    
    # prepared_df의 현재 컬럼에서 excluded_cols_for_training을 제외하여 초기 피처 컬럼 목록을 얻음
    # 이 목록은 각 종목별로 유효성 검사 후 최종 결정됨
    initial_feature_columns = [col for col in prepared_df.columns if col not in excluded_cols_for_training]


    # 각 종목별로 모델 학습 및 평가를 진행합니다.
    for ticker in tickers:
        st.write(f"--- Training model for {ticker} ---")
        
        # 특정 종목의 데이터만 필터링
        ticker_df = prepared_df[prepared_df['Ticker'] == ticker].copy()

        if ticker_df.empty:
            st.warning(f"No valid data for {ticker}. Skipping.")
            continue

        # 시계열 데이터를 위한 학습/테스트 분할
        train_size = int(len(ticker_df) * 0.8)
        train_df = ticker_df.iloc[:train_size]
        test_df = ticker_df.iloc[train_size:]

        if train_df.empty or test_df.empty:
            st.warning(f"Insufficient training or test data for {ticker}. Skipping.")
            continue

        X_train = train_df[initial_feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[initial_feature_columns]
        y_test = test_df[target_column]
        
        # 모든 피처가 숫자형인지 확인하고, NaN이 있다면 0으로 채웁니다.
        # 이 부분은 data_handler에서 처리되었어야 하지만, 최종 방어를 위해 한 번 더
        valid_feature_columns = []
        for col in initial_feature_columns:
            if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col]):
                valid_feature_columns.append(col)
            else:
                st.warning(f"Non-numeric or missing feature '{col}' in {ticker} training data. Excluding.")
                
        X_train = X_train[valid_feature_columns].fillna(0)
        X_test = X_test[valid_feature_columns].fillna(0)
        
        if ticker == tickers[0]: # 첫 번째 종목의 피처 컬럼을 대표로 저장 (모든 종목의 피처가 동일하다고 가정)
            final_feature_columns_used = valid_feature_columns

        if X_train.empty or X_test.empty:
            st.warning(f"No valid numeric features for {ticker}. Skipping model training.")
            continue


        # 모델 정의 및 학습
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        
        lgbm_model = lgb.LGBMClassifier(random_state=42)
        lgbm_model.fit(X_train, y_train)

        voting_clf_hard = VotingClassifier(estimators=[('rf', rf_model), ('lgbm', lgbm_model)], voting='hard')
        voting_clf_hard.fit(X_train, y_train)

        # 모델 평가
        y_pred_ensemble = voting_clf_hard.predict(X_test)
        
        current_ticker_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_ensemble),
            "Precision (Class 1)": precision_score(y_test, y_pred_ensemble, zero_division=0),
            "Recall (Class 1)": recall_score(y_test, y_pred_ensemble, zero_division=0),
            "F1 Score (Class 1)": f1_score(y_test, y_pred_ensemble, zero_division=0),
        }

        # ROC AUC 계산 (Soft Voting)
        try:
            voting_clf_soft = VotingClassifier(estimators=[('rf', rf_model), ('lgbm', lgbm_model)], voting='soft')
            voting_clf_soft.fit(X_train, y_train)
            if len(y_test.unique()) > 1: # ROC AUC는 두 개 이상의 클래스가 필요
                y_pred_proba_ensemble = voting_clf_soft.predict_proba(X_test)[:, 1]
                current_ticker_metrics["ROC AUC Score"] = roc_auc_score(y_test, y_pred_proba_ensemble)
            else:
                current_ticker_metrics["ROC AUC Score"] = float('nan') # 단일 클래스일 때 NaN
        except Exception as e:
            current_ticker_metrics["ROC AUC Score"] = float('nan')
            st.warning(f"Could not calculate ROC AUC for {ticker}: {e}")

        all_model_metrics[ticker] = current_ticker_metrics

        # 모델 저장
        joblib.dump(voting_clf_hard, os.path.join(MODEL_SAVE_PATH, f'ensemble_voting_classifier_hard_{ticker}.pkl'))
        joblib.dump(rf_model, os.path.join(MODEL_SAVE_PATH, f'random_forest_classifier_{ticker}.pkl'))
        joblib.dump(lgbm_model, os.path.join(MODEL_SAVE_PATH, f'lightgbm_classifier_{ticker}.pkl'))
        st.write(f"Models for {ticker} saved to {MODEL_SAVE_PATH}.")

    return all_model_metrics, final_feature_columns_used # 학습된 피처 컬럼 리스트도 함께 반환

# if __name__ == '__main__': ... (Streamlit 앱에서는 직접 호출되지 않음)
