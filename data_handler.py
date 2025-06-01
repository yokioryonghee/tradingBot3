# data_handler.py (최종 수정 버전 - app.py 파라미터 연동, fillna 강화, os import 확인)

import yfinance as yf
import pandas as pd
import numpy as np
import os # <-- os 모듈 임포트 확인
import pandas_ta as ta

# config에서 DATA_PATH와 CSV_FILENAME만 직접 임포트합니다.
# 나머지 설정값들은 함수 인수로 받습니다.
from config import DATA_PATH, CSV_FILENAME

def fetch_and_save_data_for_ticker(ticker, start_date, end_date, # start_date, end_date를 인수로 받음
                                    data_path=DATA_PATH, csv_filename_template=CSV_FILENAME):
    """
    Yahoo Finance에서 특정 주식 데이터를 가져오거나, 이미 존재하는 파일에서 로드합니다.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    current_csv_filename = csv_filename_template.format(TICKER=ticker, START_DATE=start_date, END_DATE=end_date)
    file_path = os.path.join(data_path, current_csv_filename)
    
    df = None
    should_download = True

    # 기존 파일이 있다면 삭제 후 새로 다운로드 (항상 최신 데이터 확보)
    if os.path.exists(file_path):
        print(f"기존 데이터 파일 '{file_path}'이(가) 존재합니다. 삭제 후 새로 다운로드합니다.")
        os.remove(file_path)
        should_download = True 

    if should_download:
        print(f"Yahoo Finance에서 데이터를 다운로드합니다: {ticker} ({start_date} ~ {end_date})...")
        try:
            # auto_adjust=False 명시적으로 전달하여 MultiIndex 또는 Adj Close 문제 방지
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            
            if not df.empty:
                df.index.name = 'Date'
                df.to_csv(file_path)
                print(f"데이터를 '{file_path}'에 성공적으로 저장했습니다.")
            else:
                print(f"{ticker}에 대한 데이터를 {start_date}부터 {end_date}까지 다운로드할 수 없습니다.")
                return None
        except Exception as e:
            print(f"Yahoo Finance 데이터 다운로드 중 오류 발생: {e}")
            return None
    
    # 다운로드가 완료된 후, 또는 기존 파일이 사용될 경우, 다시 로드하여 데이터프레임 확인
    # (여기서는 새로 다운로드만 하므로, 이 블록은 다운로드 실패 시 df가 None일 때 재시도)
    if df is None or df.empty:
        if os.path.exists(file_path):
            print(f"재시도: 데이터 파일 '{file_path}'을(를) 로드합니다.")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True, dayfirst=False)
                df.index = pd.to_datetime(df.index, errors='coerce')
                df.dropna(subset=[df.index.name], inplace=True)
                if df.empty:
                    raise ValueError("재로드된 CSV 파일에서 유효한 날짜 데이터를 찾을 수 없습니다.")
                if df.index.name != 'Date':
                    df.index.name = 'Date'
            except Exception as e:
                print(f"재로드 중 오류 발생 ({file_path}): {e}. 데이터 처리에 실패했습니다.")
                return None
        else:
            print("데이터 파일이 존재하지 않아 로드할 수 없습니다.")
            return None

    return df


def calculate_indicators_and_target(df_raw, ema_short_period, ema_long_period, rsi_period,
                                    macd_fast_period, macd_slow_period, macd_signal_period,
                                    prediction_horizon_days, target_rise_percent):
    """
    주어진 DataFrame에 기술적 지표와 타겟 변수를 계산하여 추가합니다.
    이 함수는 단일 종목의 데이터프레임을 처리합니다.
    """
    df = df_raw.copy()
    print(f"calculate_indicators_and_target 시작 - 초기 데이터 크기: {len(df)} 행")

    # 인덱스 확인 및 DatetimeIndex로 변환
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df.dropna(subset=[df.index.name], inplace=True) 
            if df.empty:
                print("Error: DataFrame is empty after converting index to DatetimeIndex and dropping NaT.")
                return None
        except Exception as e:
            print(f"Error converting index to DatetimeIndex in calculate_indicators_and_target: {e}")
            return None

    # 중복된 인덱스 제거
    df = df[~df.index.duplicated(keep='first')]
    print(f"지표 계산 전 (중복 제거 후) 데이터 크기: {len(df)} 행")

    # 컬럼 이름 정규화 및 MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        print(f"DEBUG: MultiIndex 컬럼을 단일 레벨로 평탄화. 새 컬럼: {df.columns.tolist()}")

    # 모든 컬럼 이름을 대문자로 통일 (YFinance 기본값)
    df.columns = [col.upper() for col in df.columns]
    print(f"DEBUG: 컬럼 이름 모두 대문자로 통일. 현재 컬럼: {df.columns.tolist()}")
    
    # 필수 컬럼이 모두 있는지 최종 확인
    expected_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    if not all(col in df.columns for col in expected_cols):
        print(f"오류: 필수 컬럼이 누락되었습니다. 필요한 컬럼: {expected_cols}, 현재 컬럼: {df.columns.tolist()}")
        return None

    # 가격 및 거래량 컬럼 데이터 타입을 숫자형으로 강제 변환
    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 변환 후 NaN이 생긴 행 제거 (필수 가격 데이터가 없는 경우)
    rows_before_price_dropna = len(df)
    df.dropna(subset=expected_cols, inplace=True)
    if len(df) < rows_before_price_dropna:
        print(f"경고: 가격/거래량 데이터 NaN 제거 후 {rows_before_price_dropna - len(df)} 행이 제거되었습니다. 현재 {len(df)} 행.")
    if df.empty:
        print("오류: 가격/거래량 NaN 제거 후 데이터가 비어 있습니다. 지표 계산 불가.")
        return None

    print(f"컬럼 이름 정규화 및 숫자형 변환 후 컬럼: {df.columns.tolist()}")
    print(f"데이터 타입 확인: \n{df[expected_cols].dtypes}")


    # 기술적 지표 계산
    df['EMA_short'] = ta.ema(df['CLOSE'], length=ema_short_period) # 인수로 받은 파라미터 사용
    print(f"DEBUG: EMA_short 계산 후 NaN 개수: {df['EMA_short'].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")

    df['EMA_long'] = ta.ema(df['CLOSE'], length=ema_long_period) # 인수로 받은 파라미터 사용
    print(f"DEBUG: EMA_long 계산 후 NaN 개수: {df['EMA_long'].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")

    df['RSI'] = ta.rsi(df['CLOSE'], length=rsi_period) # 인수로 받은 파라미터 사용
    print(f"DEBUG: RSI 계산 후 NaN 개수: {df['RSI'].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")

    # --- MACD 계산 로직 강화: 반환되는 컬럼명 패턴을 확인하여 유연하게 할당 ---
    macd_results = ta.macd(df['CLOSE'], fast=macd_fast_period, slow=macd_slow_period, signal=macd_signal_period) # 인수로 받은 파라미터 사용
    
    print(f"DEBUG: MACD 계산 결과 컬럼 (pandas_ta): {macd_results.columns.tolist()}")
    
    # MACD 라인 할당
    macd_line_found = False
    for col_name in macd_results.columns:
        if col_name.startswith('MACD_') and 'SIGNAL' not in col_name and 'HIST' not in col_name:
            df['MACD'] = macd_results[col_name]
            macd_line_found = True
            print(f"DEBUG: MACD 라인 '{col_name}'을(를) 'MACD'로 할당했습니다.")
            break
    if not macd_line_found and not macd_results.empty and len(macd_results.columns) >= 1:
        df['MACD'] = macd_results.iloc[:, 0]
        print(f"경고: MACD 라인 예상 컬럼을 찾을 수 없습니다. MACD 결과의 첫 번째 컬럼 ({macd_results.columns[0]})을 'MACD'로 할당했습니다.")
    elif not macd_line_found:
        df['MACD'] = np.nan
        print(f"오류: MACD 라인 컬럼을 찾을 수 없거나 MACD 결과가 비어있습니다. 'MACD'를 NaN으로 설정합니다.")

    # MACD 히스토그램 할당
    macd_hist_found = False
    for col_name in macd_results.columns:
        if col_name.startswith('MACDh_') or 'HIST' in col_name:
            df['MACD_hist'] = macd_results[col_name]
            macd_hist_found = True
            print(f"DEBUG: MACD 히스토그램 '{col_name}'을(를) 'MACD_hist'로 할당했습니다.")
            break
    if not macd_hist_found and not macd_results.empty and len(macd_results.columns) >= 2:
        df['MACD_hist'] = macd_results.iloc[:, 1]
        print(f"경고: MACD 히스토그램 예상 컬럼을 찾을 수 없습니다. MACD 결과의 두 번째 컬럼 ({macd_results.columns[1]})을 'MACD_hist'로 할당했습니다.")
    elif not macd_hist_found:
        df['MACD_hist'] = np.nan
        print(f"오류: MACD Histogram 컬럼을 찾을 수 없거나 MACD 결과에 충분한 컬럼이 없습니다. 'MACD_hist'를 NaN으로 설정합니다.")
                
    print(f"DEBUG: MACD/MACD_hist 계산 후 (NaN 개수: MACD={df['MACD'].isnull().sum()}, MACD_hist={df['MACD_hist'].isnull().sum()}). 현재 데이터 크기: {len(df)} 행")


    atr_col_name = f'ATR_{14}' # ATR period는 config에서 가져오지 않으므로 여기서는 14로 고정하거나 config에 추가 필요
    df[atr_col_name] = ta.atr(df['HIGH'], df['LOW'], df['CLOSE'], length=14)
    print(f"DEBUG: ATR 계산 후 NaN 개수: {df[atr_col_name].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")


    # 기타 유용한 피처
    df['return_1d'] = df['CLOSE'].pct_change()
    print(f"DEBUG: return_1d 계산 후 NaN 개수: {df['return_1d'].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")

    df['high_low_diff_pct'] = (df['HIGH'] - df['LOW']) / df['CLOSE']
    df['open_close_diff_pct'] = (df['CLOSE'] - df['OPEN']) / df['OPEN']
    df['volume_change_pct'] = df['VOLUME'].pct_change()
    
    df['close_div_ema_long'] = df['CLOSE'] / df['EMA_long'] 
    print(f"DEBUG: 기타 피처 계산 완료. close_div_ema_long NaN 개수: {df['close_div_ema_long'].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")


    print(f"모든 기술적 지표 계산 후 (NaN 포함) 총 데이터 크기: {len(df)} 행")
    
    # --- 타겟 변수 'target' 계산 로직 ---
    df['future_high'] = df['HIGH'].shift(-1).rolling(window=prediction_horizon_days, min_periods=1).max() # 인수로 받은 파라미터 사용
    print(f"DEBUG: future_high 계산 후 NaN 개수: {df['future_high'].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")

    df['target'] = ((df['future_high'] / df['CLOSE']) - 1 >= target_rise_percent).astype(int) # 인수로 받은 파라미터 사용
    print(f"DEBUG: target 계산 후 NaN 개수: {df['target'].isnull().sum()}. 현재 데이터 크기: {len(df)} 행")


    initial_rows_before_target_drop = len(df)
    df.dropna(subset=['target'], inplace=True) # 타겟 컬럼에 NaN이 있는 행 제거
    print(f"타겟 컬럼 NaN 제거 후 데이터 크기: {len(df)} 행 (제거된 NaN 행: {initial_rows_before_target_drop - len(df)})")
    
    if df.empty:
        print("경고: 타겟 컬럼 NaN 제거 후 데이터가 비어 있습니다. 이 종목은 건너뜁니다.")
        return None

    cols_to_check_for_nan = [
        'EMA_short', 'EMA_long', 'RSI', 'MACD', 'MACD_hist', atr_col_name,
        'return_1d', 'high_low_diff_pct', 'open_close_diff_pct', 
        'volume_change_pct', 'close_div_ema_long', 'target',
    ]
    
    for col in cols_to_check_for_nan:
        if col not in df.columns:
            print(f"오류: 최종 NaN 검사 시 필수 컬럼 '{col}'이(가) DataFrame에 없습니다. 데이터 처리에 실패했습니다.")
            return None

    rows_before_fillna = len(df)
    for col in cols_to_check_for_nan:
        if df[col].isnull().any():
            nan_count = df[col].isnull().sum()
            print(f"DEBUG: {col} 컬럼에 {nan_count}개의 NaN이 있습니다. fillna(method='ffill') 적용.")
            df[col].fillna(method='ffill', inplace=True)
            if df[col].isnull().any():
                print(f"DEBUG: {col} 컬럼에 ffill 후에도 {df[col].isnull().sum()}개의 NaN이 남아있습니다. fillna(method='bfill') 적용.")
                df[col].fillna(method='bfill', inplace=True)
            if df[col].isnull().any():
                print(f"DEBUG: {col} 컬럼에 bfill 후에도 {df[col].isnull().sum()}개의 NaN이 남아있습니다. 0으로 채우기 적용.")
                df[col].fillna(0, inplace=True)

    print(f"fillna() 처리 후 최종 데이터 크기: {len(df)} 행")

    for col in cols_to_check_for_nan:
        if df[col].isnull().any():
            print(f"오류: fillna() 후에도 컬럼 '{col}'에 NaN이 남아있습니다. ({df[col].isnull().sum()}개)")
            df.dropna(subset=[col], inplace=True)
            print(f"경고: '{col}'의 남은 NaN 행 제거 후 데이터 크기: {len(df)} 행")

    if df.empty:
        print("경고: 결측치 처리 후 데이터가 비어 있습니다. 데이터 기간 또는 파라미터(예: EMA 기간, PREDICTION_HORIZON_DAYS)를 확인하십시오.")
        return None

    return df

# get_prepared_data_for_multiple_tickers 함수 시그니처 변경!
def get_prepared_data(tickers, start_date_str, end_date_str, 
                                            ema_short_period, ema_long_period, rsi_period,
                                            macd_fast_period, macd_slow_period, macd_signal_period,
                                            prediction_horizon_days, target_rise_percent):
    """
    여러 종목에 대해 데이터를 가져와 기술적 지표와 타겟 변수를 계산하고,
    모든 데이터를 하나의 DataFrame으로 합쳐 반환하는 메인 함수.
    모든 설정값을 인수로 받도록 변경.
    """
    all_data = []
    print(f"데이터 준비 중: {len(tickers)}개 종목 ({start_date_str} ~ {end_date_str})")

    for ticker in tickers:
        print(f"\n--- {ticker} 데이터 처리 시작 ---")
        df_raw = fetch_and_save_data_for_ticker(ticker, start_date=start_date_str, end_date=end_date_str)
        if df_raw is None or df_raw.empty:
            print(f"{ticker} 원시 데이터 로드 실패. 다음 종목으로 넘어갑니다.")
            continue
        
        df_with_features_and_target = calculate_indicators_and_target(
            df_raw,
            ema_short_period=ema_short_period,
            ema_long_period=ema_long_period,
            rsi_period=rsi_period,
            macd_fast_period=macd_fast_period,
            macd_slow_period=macd_slow_period,
            macd_signal_period=macd_signal_period,
            prediction_horizon_days=prediction_horizon_days,
            target_rise_percent=target_rise_percent
        )
        
        if df_with_features_and_target is None or df_with_features_and_target.empty:
            print(f"{ticker} 지표/타겟 계산 후 데이터 없음. 다음 종목으로 넘어갑니다.")
            continue
        
        df_with_features_and_target['Ticker'] = ticker
        all_data.append(df_with_features_and_target)
        print(f"--- {ticker} 데이터 처리 완료. 최종 유효 데이터 크기: {len(df_with_features_and_target)} 행 ---")

    if not all_data:
        print("모든 종목에 대해 데이터 준비에 실패했습니다.")
        return None
    
    final_df = pd.concat(all_data).sort_index()
    
    print(f"\n모든 종목 지표 및 타겟 계산 완료. 최종 유효 데이터 크기: {len(final_df)} 행")
    return final_df

# `if __name__ == '__main__':` 블록은 `app.py`에서 호출될 때 실행되지 않습니다.
# 하지만 단독 테스트를 위해 config 값들을 임포트하여 인수로 전달합니다.
if __name__ == '__main__':
    print("--- data_handler.py 단독 실행 테스트 시작 ---")
    from config import ( # config에서 모든 필요한 변수 임포트
        STOCK_UNIVERSE, START_DATE, END_DATE, 
        EMA_SHORT_PERIOD, EMA_LONG_PERIOD, RSI_PERIOD,
        MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
        PREDICTION_HORIZON_DAYS, TARGET_RISE_PERCENT
    )

    prepared_df_all_tickers = get_prepared_data( # 함수명 변경
        tickers=STOCK_UNIVERSE,
        start_date_str=START_DATE,
        end_date_str=END_DATE,
        ema_short_period=EMA_SHORT_PERIOD,
        ema_long_period=EMA_LONG_PERIOD,
        rsi_period=RSI_PERIOD,
        macd_fast_period=MACD_FAST_PERIOD,
        macd_slow_period=MACD_SLOW_PERIOD,
        macd_signal_period=MACD_SIGNAL_PERIOD,
        prediction_horizon_days=PREDICTION_HORIZON_DAYS,
        target_rise_percent=TARGET_RISE_PERCENT
    ) 

    if prepared_df_all_tickers is not None and not prepared_df_all_tickers.empty:
        print("\n--- 준비된 전체 데이터 (처음 5행) ---")
        print(prepared_df_all_tickers.head())
        
        print(f"\n--- 준비된 전체 데이터 기간: {prepared_df_all_tickers.index.min()} ~ {prepared_df_all_tickers.index.max()} ---")
        print(f"총 데이터 행 수: {len(prepared_df_all_tickers)}")
        print(f"총 컬럼 수: {len(prepared_df_all_tickers.columns)}")
        print(f"포함된 종목: {prepared_df_all_tickers['Ticker'].unique().tolist()}")
        
        print("\n--- 각 종목별 타겟 분포 ---")
        print(prepared_df_all_tickers.groupby('Ticker')['target'].value_counts())
        
    else:
        print("데이터 준비에 실패했습니다 (메인 함수).")
