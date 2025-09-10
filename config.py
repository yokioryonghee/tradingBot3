# config.py

# 데이터 관련 설정
STOCK_UNIVERSE = [
    'SPY', 'AAPL', 'GOOGL', 'MSFT', 'SOXL',
    'NVDA', 'TSLA', 'QQQS', 'TSLQ', 'TSLS',
    'BITO', 'JEPI', 'JEPQ'
]  # 다양한 종목 포함 (테스트용)

# 기본 날짜값 (Streamlit에서 초기값으로만 사용됨)
DEFAULT_START_DATE = '2018-01-01'
DEFAULT_END_DATE = '2025-12-31'

DATA_PATH = 'data/'  # 로컬에서 데이터 저장 시 사용될 폴더
CSV_FILENAME = "{TICKER}_{START_DATE}_to_{END_DATE}_new.csv"

# ML 모델에서 사용할 피처
ML_MODEL_FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI', 'MACD', 'BollingerBands_Upper', 'BollingerBands_Lower',
    'EMA_Short', 'EMA_Long', 'ADX', 'momentum',
    'volume_change_pct', 'test_size'
]

# 기술적 지표 파라미터
EMA_SHORT_PERIOD = 20
EMA_LONG_PERIOD = 60
RSI_PERIOD = 21
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# 백테스팅 관련 설정
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_PERCENT = 0.001
MAX_ALLOCATION_PCT_OF_INITIAL_CAPITAL_PER_TRADE = 0.1

TAKE_PROFIT_PERCENT = 0.07  # 목표 수익률 7%
STOP_LOSS_PERCENT = 0.03    # 손절매 3%
MAX_HOLD_DAYS = 365
MAX_CONCURRENT_POSITIONS = 6

# 타겟 변수 계산을 위한 설정
PREDICTION_HORIZON_DAYS = 20
TARGET_RISE_PERCENT = 0.10

# RSI 트렌드 확인 레벨 (전략에 사용)
RSI_TREND_CONFIRM_LEVEL_BUY = 55
RSI_TREND_CONFIRM_LEVEL_SELL = 45

TEST_SIZE = 20
RANDOM_STATE = 42
