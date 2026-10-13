from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_default_ui_loads_and_missing_data_is_visible():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    app.button[1].click().run(timeout=30)
    assert not app.exception and app.error


def test_ui_training_and_backtest_flow():
    from test_regressions import bars
    from technical_indicators import prepare_features
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    app.text_input[0].set_value('A')
    next(s for s in app.slider if s.label=='Prediction horizon (sessions)').set_value(5)
    next(s for s in app.slider if s.label=='Target rise (%)').set_value(1.)
    app.run(timeout=30)
    data=prepare_features(bars(650),horizon=5,rise=.01); data['Ticker']='A'
    app.session_state['prepared_data']=data
    app.button[1].click().run(timeout=60)
    assert not app.exception and not app.error
    app.radio[0].set_value('ML-Based').run(timeout=30)
    app.button[2].click().run(timeout=60)
    assert not app.exception and not app.error
