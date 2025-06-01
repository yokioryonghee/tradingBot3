# performance_analyzer.py (Streamlit 연동을 위해 plot_performance 함수 변경)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from config import TICKER # Streamlit 앱에서 직접 티커를 받으므로 여기서는 사용 안 함

def calculate_performance_metrics(portfolio_df, initial_capital):
    """포트폴리오 DataFrame으로부터 주요 성과 지표를 계산합니다."""
    if portfolio_df.empty:
        return {
            "Total Return (%)": 0, "CAGR (%)": 0,
            "Max Drawdown (%)": 0, "Sharpe Ratio": 0,
            "Number of Trades": 0 
        }

    # 인덱스가 DatetimeIndex가 아니면 변환
    if not isinstance(portfolio_df.index, pd.DatetimeIndex):
        try:
            portfolio_df.index = pd.to_datetime(portfolio_df.index)
        except Exception as e:
            print(f"Error converting portfolio index to DatetimeIndex: {e}")
            return {
                "Total Return (%)": 0, "CAGR (%)": 0,
                "Max Drawdown (%)": 0, "Sharpe Ratio": 0,
                "Number of Trades": 0
            }

    final_value = portfolio_df['total_value'].iloc[-1]
    total_return_pct = (final_value / initial_capital - 1) * 100
    
    # CAGR 계산
    if len(portfolio_df.index) > 1:
        num_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        num_years = num_days / 365.25
    else:
        num_years = 0
        
    cagr = 0
    if num_years > 0 and initial_capital > 0 and final_value > 0:
         cagr = ((final_value / initial_capital) ** (1 / num_years) - 1) * 100
    
    # Max Drawdown (MDD) 계산
    portfolio_df['peak'] = portfolio_df['total_value'].cummax()
    portfolio_df['drawdown'] = portfolio_df['total_value'] - portfolio_df['peak']
    portfolio_df['drawdown_pct'] = (portfolio_df['drawdown'] / portfolio_df['peak']) * 100
    max_drawdown_pct = portfolio_df['drawdown_pct'].min() 
    
    # Sharpe Ratio 계산
    daily_returns = portfolio_df['returns']
    sharpe_ratio = 0
    if not daily_returns.empty and daily_returns.std() != 0: 
         sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) # 252 거래일 기준
    
    metrics = {
        "Total Return (%)": round(total_return_pct, 2),
        "CAGR (%)": round(cagr, 2),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2)
    }
    return metrics

def plot_performance(portfolio_df, stock_data_for_benchmark, ticker_name): # ticker -> ticker_name으로 변경
    """
    포트폴리오 가치 변화와 벤치마크(원시 주가)를 함께 시각화합니다.
    matplotlib.figure.Figure 객체를 반환하여 Streamlit에 표시합니다.
    """
    fig, ax = plt.subplots(figsize=(12, 7)) # Figure와 Axes 객체를 함께 생성

    # 포트폴리오 가치 플로팅
    ax.plot(portfolio_df.index, portfolio_df['total_value'], label='Portfolio Value', color='blue')

    # 벤치마크 (원시 주가) 플로팅
    if not portfolio_df.empty:
        plot_start_date = portfolio_df.index.min()
        benchmark_aligned = stock_data_for_benchmark['CLOSE'].copy() 
    else:
        print("경고: 포트폴리오 데이터가 비어있어 벤치마크를 그릴 수 없습니다.")
        plt.close(fig) # 생성된 Figure 닫기
        return None

    if benchmark_aligned.empty:
        print("경고: 벤치마크 데이터가 필터링된 후 비어있습니다. 벤치마크는 그려지지 않습니다.")
    else:
        initial_benchmark_price = benchmark_aligned.iloc[0]
        scaled_benchmark = portfolio_df['total_value'].iloc[0] * (benchmark_aligned / initial_benchmark_price)
        ax.plot(scaled_benchmark.index, scaled_benchmark, label=f'{ticker_name} Buy & Hold (Scaled)', color='orange', linestyle='--')

    ax.set_title(f'Portfolio Performance vs. {ticker_name} Buy & Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout() # tight_layout은 Figure 객체에서 호출

    # X축 범위를 백테스팅 데이터 기간에 맞춥니다.
    if not portfolio_df.empty:
        ax.set_xlim(portfolio_df.index.min(), portfolio_df.index.max())

    # Y축 스케일을 자동으로 최적화하도록 합니다.
    # ax.set_ylim(portfolio_df['total_value'].min() * 0.9, portfolio_df['total_value'].max() * 1.1)
    
    return fig # Figure 객체를 반환

# if __name__ == '__main__': ... (Streamlit 앱에서는 사용 안 함)
