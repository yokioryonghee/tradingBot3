# strategy.py

import pandas as pd
# 필요한 다른 임포트 (예: pandas_ta를 사용한다면 import pandas_ta as ta)

def generate_signals(df_with_indicators):
    """
    규칙 기반 매매 신호를 생성합니다.
    Args:
        df_with_indicators (pd.DataFrame): 기술 지표가 포함된 데이터프레임.
                                          'EMA_Short', 'EMA_Long', 'RSI', 'MACD_Signal_Line' 등
                                          필요한 지표 컬럼을 포함해야 합니다.
    Returns:
        pd.DataFrame: 'Date'를 인덱스로 하고 'signal' 컬럼을 가진 DataFrame.
                      signal: 1 (매수), -1 (매도), 0 (관망)
    """
    signals = pd.DataFrame(index=df_with_indicators.index)
    signals['signal'] = 0 # 기본값은 관망 (0)

    # 매수 신호 조건 (예시: 단기 EMA가 장기 EMA를 상향 돌파하고 RSI가 과매도 구간이 아닐 때)
    buy_condition = (
        (df_with_indicators['EMA_Short'] > df_with_indicators['EMA_Long']) &
        (df_with_indicators['EMA_Short'].shift(1) <= df_with_indicators['EMA_Long'].shift(1)) &
        (df_with_indicators['RSI'] < 70)
    )
    signals.loc[buy_condition, 'signal'] = 1

    # 매도 신호 조건 (예시: 단기 EMA가 장기 EMA를 하향 돌파할 때)
    sell_condition = (
        (df_with_indicators['EMA_Short'] < df_with_indicators['EMA_Long']) &
        (df_with_indicators['EMA_Short'].shift(1) >= df_with_indicators['EMA_Long'].shift(1))
    )
    signals.loc[sell_condition, 'signal'] = -1

    if 'Ticker' in df_with_indicators.columns:
        signals['Ticker'] = df_with_indicators['Ticker']
    else:
        signals['Ticker'] = 'UNKNOWN'

    return signals
