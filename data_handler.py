# data_handler.py (RSI 적용 및 구조 정리 버전)

import yfinance as yf
import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from config import DATA_PATH, CSV_FILENAME

def fetch_and_save_data_for_ticker(ticker, start_date, end_date,
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

    if os.path.exists(file_path):
        print(f"기존 데이터 파일 '{file_path}'이(가) 존재합니다. 삭제 후 새로 다운로드합니다.")
        os.remove(file_path)
        should_download = True

    if should_download:
        print(f"Yahoo Finance에서 데이터를 다운로드합니다: {ticker} ({start_date} ~ {end_date})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

            if not df.empty:
                df.index.name = 'Date'

                # ✅ RSI 계산 (Close 컬럼 기준)
                df['rsi'] = RSIIndicator(close=df['Close']).rsi()

                df.to_csv(file_path)
                print(f"데이터를 '{file_path}'에 성공적으로 저장했습니다.")
            else:
                print(f"{ticker}에 대한 데이터를 {start_date}부터 {end_date}까지 다운로드할 수 없습니다.")
                return None
        except Exception as e:
            print(f"Yahoo Finance 데이터 다운로드 중 오류 발생: {e}")
            return None

    return df
