# 📈 Trading Bot App

이 프로젝트는 머신러닝 기반 전략을 Streamlit 대시보드로 시각화하고, 백테스트하는 트레이딩 앱입니다.

## 🚀 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 주요 파일 구조

- `app.py` : Streamlit 앱 엔트리포인트
- `backtester.py`, `strategy.py` : 백테스트 로직
- `ml_strategy.py`, `train_evaluate_ml.py` : ML 전략과 학습
- `performance_analyzer.py` : 성과 지표 계산
- `data_handler.py` : 데이터 로딩 및 전처리

---

👉 Streamlit Cloud 또는 GitHub에 배포하여 쉽게 실행할 수 있습니다.