from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_default_ui_loads_and_missing_data_is_visible():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    app.button[1].click().run(timeout=30)
    assert not app.exception and app.error
