# Daily-bar strategy research

Streamlit에서 조정주가를 내려받고 규칙 전략과 RF/LightGBM 분류 전략을 비교합니다. 신호는 종가 이후 계산하고 다음 관측 시가에 체결합니다.

## 🚀 실행 방법

이 명령은 **tradingBot3** 프로젝트용입니다. `quantitative-trading-engine` 폴더에서는 실행하지 마세요. `app.py`와 `requirements.txt`가 있는 저장소 최상위 폴더에서 실행해야 합니다. Python 3.11을 권장합니다.

### macOS / Linux — 처음 설치

아직 저장소가 없다면 다음 명령으로 내려받습니다.

```bash
cd ~
git clone https://github.com/yokioryonghee/tradingBot3.git
cd tradingBot3
```

이미 내려받았다면 위 clone 명령 대신 기존 `tradingBot3` 폴더로 이동하세요.

Python 3.11이 설치되어 있다면 다음을 실행합니다.

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py
```

**pyenv를 사용하는 경우:** 저장소의 `.python-version`에 적힌 환경 이름이 내 컴퓨터에 없을 수 있습니다. 위의 가상환경 생성 명령 대신 아래 두 줄을 실행한 후, 위의 설치·실행 명령 세 줄을 실행하세요.

```bash
pyenv install -s 3.11.8
PYENV_VERSION=3.11.8 python -m venv .venv
```

### macOS / Linux — 다음부터 실행

`tradingBot3` 폴더에서 다음 한 줄로 실행합니다. 가상환경 활성화는 필요하지 않습니다.

```bash
./.venv/bin/python -m streamlit run app.py
```

터미널에 표시되는 Local URL(기본값: http://localhost:8501)을 브라우저에서 엽니다. 종료는 터미널에서 `Ctrl+C`를 누릅니다.

### Windows PowerShell

Python 3.11 설치 후 `tradingBot3` 폴더에서 실행합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

다음부터는 마지막 실행 명령만 사용합니다.

### 실행 오류 확인

- `pyenv: streamlit: command not found`: 위처럼 `./.venv/bin/python -m streamlit run app.py`로 실행하세요. 설치와 실행이 같은 Python을 사용합니다.
- `No module named streamlit`: 프로젝트 폴더에서 `./.venv/bin/python -m pip install -r requirements.txt`를 다시 실행하세요.
- `File does not exist: app.py`: 현재 폴더가 `tradingBot3`인지 확인하세요.
- `Requirement already satisfied`: 패키지가 이미 설치되었다는 정상 메시지입니다.


## 계산 방식

- 지표는 현재와 과거 관측만 사용합니다. 초기 지표 구간은 제거하고 미래 값으로 채우지 않습니다.
- 상승 타깃은 다음 h개 관측의 최고가로 계산합니다. 아직 끝나지 않은 미래 라벨은 결측으로 유지합니다.
- 종목별로 신호를 계산합니다. 오늘 종가로 생성한 주문은 이후 해당 종목의 관측 시가에 실행합니다.
- 익절·손절은 종가에서 관찰하고 다음 관측 시가에 청산합니다. 일중 가격 경로와 보장된 손절 체결은 모델링하지 않습니다.
- 현금, 정수 주식 수량과 편도 수수료를 추적합니다. 휴장일은 마지막 관측 가격으로 평가하고, 데이터 구간 종료 시 각 종목의 마지막 관측 종가로 청산합니다.
- ML 학습/검증은 시간 순서로 나누고 라벨이 다음 구간을 침범하는 표본을 제외합니다. 모델의 feature schema와 마지막 validation label 날짜를 metadata에 저장합니다.
- ML 백테스트는 학습 및 보고된 검증 구간이 모두 끝난 이후 날짜만 사용합니다. 모델이 없거나 feature schema가 맞지 않으면 오류를 표시합니다.

## UI 사용

1. 종목, 날짜와 지표 설정을 선택하고 Prepare data를 누릅니다.
2. ML을 사용할 경우 Train ML models로 앞쪽 데이터에서 학습/검증합니다.
3. Rules-Based 또는 ML-Based를 선택하고 Run backtest를 누릅니다.

데이터 설정을 바꾸면 이전 준비 데이터와 학습 상태를 무효화합니다. 새 설정으로 다시 준비·학습하세요.

## 테스트

```bash
python -m pip install pytest
python -m pytest -q
```

회귀 테스트는 미래 최고가 라벨, 지표 인과성, 종목 간 신호 분리, 다음 시가 체결, 최종 수수료, 휴장일 평가와 ML 구간 분리를 검사합니다. 시세 다운로드는 mock으로 검사합니다.

## 범위

일별 연구 시뮬레이션입니다. 실시간 데이터, 주문 전송, 부분 체결, 계좌 동기화, 시장 충격, 차입, 환전은 포함하지 않습니다. 실거래 수익이나 미래 예측 우위는 입증하지 않았습니다. 현재 예제 유니버스는 선택된 미국 종목이며 생존 편향 없는 시장 전체 데이터가 아닙니다.
