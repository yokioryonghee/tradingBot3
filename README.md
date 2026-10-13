
https://tradingbot3-aacevydlbvfcxoqpfal5nj.streamlit.app

if you want to try it by yourself, go to [tradingbot repo](https://github.com/yokioryonghee/trading)

Quantitative Trading Bot with Machine Learning & Interactive Visualization
Project Overview
This project, tradingBot3, is a comprehensive application designed for developing, backtesting, and visualizing quantitative trading strategies, with a strong emphasis on Machine Learning (ML) for signal generation. It demonstrates a full end-to-end pipeline for algorithmic trading strategy development, from data acquisition and preprocessing to advanced analytical modeling, performance evaluation, and interactive reporting.
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/0728c886-c070-4ebf-a827-7046c507d808" />

Key Features
Robust Data Handling: Efficiently collects and preprocesses historical stock data, integrating essential technical indicators for strategic analysis.

Machine Learning-Driven Strategy: Leverages ML models to generate intelligent buy/hold signals, moving beyond traditional rule-based approaches.

Comprehensive Backtesting Engine: Simulates trading performance on historical data, allowing for rigorous validation of strategy profitability and risk.

In-depth Performance Analysis: Calculates critical financial metrics such as total return, annualized return, maximum drawdown (MDD), and Sharpe Ratio to provide a quantitative assessment of strategy effectiveness.

Interactive Visualization with Streamlit: Presents complex backtesting results and portfolio evolution through an intuitive, interactive web dashboard, enhancing interpretability for both technical and non-technical stakeholders.

Technical Stack
Programming Language: Python

Data Manipulation: Pandas, NumPy

Machine Learning: TensorFlow/Keras (for model training and prediction)

Web Application/Visualization: Streamlit

System Utilities: argparse (for command-line interface), watchdog (for real-time file system monitoring)

Project Architecture & Workflow
The tradingBot3 system is structured into modular components, ensuring clarity, maintainability, and scalability. The workflow encompasses several distinct phases:

Data Acquisition & Preparation:
Historical stock data is fetched and enriched with relevant technical indicators.
[데이터 핸들링 모듈의 코드 스크린샷 또는 데이터프레임 예시 스크린샷]

Machine Learning Model Training:
A dedicated module handles the training of an ML model (e.g., MobileNetV2 for transfer learning, or a custom model) using pre-labeled historical data. This model learns patterns to predict future price movements or optimal trading actions.
[ML 모델 학습 코드 스크린샷 또는 학습 과정 출력 스크린샷]

ML-Driven Signal Generation:
The trained ML model is applied to new or unseen data to generate buy (1.0) or hold (0.0) signals. This forms the core of the algorithmic trading strategy.
[ML 신호 생성 코드 스크린샷 또는 신호가 추가된 데이터프레임 예시 스크린샷]

Backtesting Simulation:
The generated trading signals are fed into a robust backtesting engine. This simulates trades based on historical prices, accounting for initial capital and calculating portfolio value over time.
[백테스팅 로직의 핵심 코드 스크린샷 또는 백테스팅 진행 출력 스크린샷]

Performance Analysis & Reporting:
Post-backtesting, a performance analyzer calculates key financial metrics to objectively evaluate the strategy's profitability and risk-adjusted returns.
<img width="201" height="103" alt="Image" src="https://github.com/user-attachments/assets/d6d49d1a-3765-4917-8ef8-e2400f8f0b22" />

Interactive Visualization:
All results, including portfolio equity curves, trade history, and performance metrics, are presented via a Streamlit web application, offering an interactive and user-friendly interface for analysis.
[Streamlit 대시보드 전체 스크린샷]
[Streamlit 대시보드의 포트폴리오 곡선 스크린샷]
[Streamlit 대시보드의 성과 지표 요약 스크린샷]

Continuous Monitoring (Optional/Future):
The system can be extended to continuously monitor incoming data, automatically triggering the classification and saving process for new information.
[지속적인 모니터링 코드 스크린샷 또는 실행 중인 콘솔 스크린샷]

Strengths & Why This Project Matters
This project serves as a compelling demonstration of my capabilities in several critical areas:

End-to-End System Development: I have successfully implemented a complete quantitative trading workflow, from raw data to actionable insights and interactive reporting. This showcases my ability to design, build, and integrate complex systems.

Applied Machine Learning: My proficiency in applying ML techniques to real-world financial problems is evident. This project highlights my understanding of ML model lifecycle, feature engineering, and predictive modeling for decision-making.

Strong Software Engineering Practices: The modularized code structure (e.g., separate modules for data, ML, backtesting, performance) reflects a commitment to clean, maintainable, reusable, and scalable code. This is crucial for collaborative development and long-term project health.

Quantitative Analysis & Financial Acumen: The robust backtesting and comprehensive performance analysis (MDD, Sharpe Ratio, etc.) demonstrate my understanding of financial metrics and my ability to rigorously validate trading strategies.

Effective Communication & Visualization: Leveraging Streamlit to create an interactive dashboard underscores my ability to translate complex analytical results into easily digestible visual formats, facilitating communication with diverse audiences.

Practical Python & Library Proficiency: The project leverages industry-standard Python libraries (Pandas, NumPy, TensorFlow/Keras, Streamlit), showcasing practical skills highly valued in data science and FinTech roles.

Proactive Problem Solving & Learning: Undertaking such a multifaceted project independently highlights my initiative and my rapid learning curve in acquiring and applying new technologies and concepts.

Future Enhancements & Next Steps
While this project provides a solid foundation, I am continuously looking for opportunities to enhance its capabilities and robustness:

Deepen ML Model Explainability: Implement techniques (e.g., SHAP, LIME) to understand why the ML model makes specific trading decisions, improving trust and interpretability.

Enhance Backtesting Realism: Incorporate more realistic trading costs (commissions, taxes), slippage models, and market impact considerations into the backtester.py to provide a more accurate reflection of live trading conditions.

Robust Data Pipeline: Strengthen error handling and data validation within data_handler.py to gracefully manage API failures, missing data, and data inconsistencies. Explore more advanced data cleaning and imputation techniques.

Comprehensive Code Documentation: Further enrich the internal code with detailed docstrings for all functions and classes, and add more inline comments to explain complex logic, ensuring maximum clarity for future development.

Unit Testing Framework: Implement unit tests using pytest for critical modules (e.g., data handling, signal generation, performance calculation) to ensure code reliability and facilitate future modifications.

Live Trading Integration (Exploration): Research and plan for potential integration with brokerage APIs for real-time data feeds and order execution, transitioning from backtesting simulation to potential live trading (with appropriate risk management considerations).

Expand Strategy Complexity: Explore incorporating reinforcement learning, natural language processing (for sentiment analysis), or multi-asset allocation strategies.

Conclusion
This tradingBot3 project represents my passion for combining quantitative analysis, machine learning, and software development to solve complex financial challenges. It demonstrates not only my technical proficiency across various domains but also my ability to build a functional, well-structured system from conception to interactive deployment. I am confident that the skills honed through this project would enable me to make immediate and valuable contributions to your team.








# 📈 Trading Bot App

이 프로젝트는 머신러닝 기반 전략을 Streamlit 대시보드로 시각화하고, 백테스트하는 트레이딩 앱입니다.

## 🚀 실행 방법

이 명령은 **tradingBot3** 프로젝트용입니다. `quantitative-trading-engine` 폴더에서는 실행하지 마세요. `app.py`와 `requirements.txt`가 있는 저장소 최상위 폴더에서 실행해야 합니다. Python 3.11을 권장합니다.

### macOS / Linux — 처음 설치

아직 저장소가 없다면 다음 명령으로 내려받습니다.

```bash
cd ~
git clone --branch fix/reproducible-runtime-20261013 https://github.com/yokioryonghee/tradingBot3.git
cd tradingBot3
```

현재 실행 가능한 수정본은 `fix/reproducible-runtime-20261013` 브랜치에 있습니다. 기존 `main`에는 Python 3.11에서 설치되지 않는 `pandas_ta` 의존성이 남아 있으므로 아래 브랜치 전환이 필요합니다.

이미 내려받았다면 기존 `tradingBot3` 폴더에서 다음을 실행하세요. 로컬 변경 때문에 전환이 거부되면 변경 내용을 보존한 뒤 진행하세요.

```bash
git fetch origin
git switch fix/reproducible-runtime-20261013
git pull --ff-only origin fix/reproducible-runtime-20261013
```

기존 Python 3.11 가상환경이 있다면 다시 생성할 필요 없이 아래 패키지 설치부터 진행합니다.

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
- `No matching distribution found for pandas_ta`: 위의 수정 브랜치로 전환한 뒤 설치 명령을 다시 실행하세요. 수정본은 `pandas_ta` 없이 지표를 계산합니다.
- `No module named streamlit`: 프로젝트 폴더에서 `./.venv/bin/python -m pip install -r requirements.txt`를 다시 실행하세요.
- `File does not exist: app.py`: 현재 폴더가 `tradingBot3`인지 확인하세요.
- `Requirement already satisfied`: 패키지가 이미 설치되었다는 정상 메시지입니다.

## 📁 주요 파일 구조

- `app.py` : Streamlit 앱 엔트리포인트
- `backtester.py`, `strategy.py` : 백테스트 로직
- `ml_strategy.py`, `train_evaluate_ml.py` : ML 전략과 학습
- `performance_analyzer.py` : 성과 지표 계산
- `data_handler.py` : 데이터 로딩 및 전처리

---

👉 Streamlit Cloud 또는 GitHub에 배포하여 쉽게 실행할 수 있습니다.
