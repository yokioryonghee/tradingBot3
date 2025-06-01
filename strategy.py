# strategy.py

import pandas as pd
import pandas_ta as ta # 기술적 지표 계산을 위해 필요

def generate_signals(df_with_indicators, ema_short_period, ema_long_period, rsi_period, rsi_buy_level, rsi_sell_level):
    """
    규칙 기반 매매 신호를 생성합니다.
    Args:
        df_with_indicators (pd.DataFrame): 기술 지표가 포함된 데이터프레임.
                                          'EMA_Short', 'EMA_Long', 'RSI' 등의 컬럼을 포함해야 합니다.
        ema_short_period (int): 단기 EMA 기간
        ema_long_period (int): 장기 EMA 기간
        rsi_period (int): RSI 기간
        rsi_buy_level (int): RSI 매수 기준 레벨
        rsi_sell_level (int): RSI 매도 기준 레벨
    Returns:
        pd.DataFrame: 'Date'를 인덱스로 하고 'signal'과 'Ticker' 컬럼을 가진 DataFrame.
                      signal: 1 (매수), -1 (매도), 0 (관망)
    """
    signals = pd.DataFrame(index=df_with_indicators.index)
    signals['signal'] = 0 # 기본값은 관망 (0)

    # 지표가 이미 계산되어 있다고 가정하지만, 혹시 없을 경우를 대비하여 재계산 로직 추가 (선택 사항)
    # 실제 앱에서는 data_handler에서 계산된 데이터를 받으므로 여기서는 생략 가능

    # 매수 신호 조건: 단기 EMA가 장기 EMA를 상향 돌파하고 RSI가 특정 레벨 이하일 때
    # df_with_indicators에 이미 'EMA_short', 'EMA_long', 'RSI' 컬럼이 있다고 가정
    buy_condition = (
        (df_with_indicators['EMA_short'] > df_with_indicators['EMA_long']) &
        (df_with_indicators['EMA_short'].shift(1) <= df_with_indicators['EMA_long'].shift(1)) &
        (df_with_indicators['RSI'] < rsi_buy_level) # RSI 매수 레벨 사용
    )
    signals.loc[buy_condition, 'signal'] = 1

    # 매도 신호 조건: 단기 EMA가 장기 EMA를 하향 돌파하고 RSI가 특정 레벨 이상일 때
    sell_condition = (
        (df_with_indicators['EMA_short'] < df_with_indicators['EMA_long']) &
        (df_with_indicators['EMA_short'].shift(1) >= df_with_indicators['EMA_long'].shift(1)) &
        (df_with_indicators['RSI'] > rsi_sell_level) # RSI 매도 레벨 사용
    )
    signals.loc[sell_condition, 'signal'] = -1

    # 각 종목별로 신호가 생성되도록 'Ticker' 컬럼을 추가
    if 'Ticker' in df_with_indicators.columns:
        signals['Ticker'] = df_with_indicators['Ticker']
    else:
        # Ticker 컬럼이 없는 경우 (단일 종목 데이터라고 가정), 기본값 설정
        signals['Ticker'] = df_with_indicators.index.get_level_values('Ticker') if isinstance(df_with_indicators.index, pd.MultiIndex) else 'UNKNOWN'

    # 신호 DataFrame 정리 (신호가 0인 행은 필요 없는 경우 제거 가능)
    # signals = signals[signals['signal'] != 0]

    return signals