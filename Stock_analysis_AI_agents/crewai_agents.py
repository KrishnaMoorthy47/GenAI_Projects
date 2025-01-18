import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

GROQ_API_KEY = ""

COMMON_STOCKS = {
    'TCS': 'TCS.NS',
    'INFOSYS': 'INFY.NS',
    'RELIANCE': 'RELIANCE.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'SBI': 'SBIN.NS',
    'WIPRO': 'WIPRO.NS',
    'BAJAJ FINANCE': 'BAJFINANCE.NS',
    'MARUTI SUZUKI': 'MARUTI.NS',
    'AXIS BANK': 'AXISBANK.NS'
}

st.set_page_config(page_title="Indian Stocks Analysis with CrewAI", page_icon="", layout="wide")

# CSS Styling 
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stApp { max-width: 1400px; margin: 0 auto; }
    .card {
        background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e1e4e8;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0366d6;
    }
    .metric-label {
        font-size: 14px;
        color: #586069;
        text-transform: uppercase;
    }
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e1e4e8;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_agents(symbol):
    """
    Initialize crewAI agents for stock analysis
    """
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="claude-3-5-sonnet-20240620"
        )
        
        # Creating Stock Analyst Agent
        stock_analyst = Agent(
            role='Stock Market Analyst',
            goal=f'Provide comprehensive and insightful analysis of {symbol} stock performance',
            backstory='An experienced financial analyst with deep understanding of market trends and company fundamentals',
            verbose=True,
            llm=llm
        )
        
        # Creating Company Research Agent
        company_researcher = Agent(
            role='Company Researcher',
            goal=f'Gather detailed information about {symbol} company\'s financial health and market position',
            backstory='A meticulous researcher specializing in company background, financial statements, and market positioning',
            verbose=True,
            llm=llm
        )
        
        # Creating a task for stock analysis
        stock_analysis_task = Task(
            description=f'Analyze the {symbol} stock performance, provide key insights, and assess potential investment opportunities',
            agent=stock_analyst,
            expected_output=f"""A comprehensive analysis of {symbol} stock, including:
            1. Current market performance
            2. Key financial metrics
            3. Investment recommendation
            4. Short-term and long-term outlook
            5. Potential risks and opportunities"""
        )
        
        # Creating a task for company research
        company_research_task = Task(
            description=f'Conduct in-depth research on {symbol} company\'s background, financial health, and market position',
            agent=company_researcher,
            expected_output=f"""Detailed company research report for {symbol}, covering:
            1. Company background and history
            2. Business model and key revenue streams
            3. Competitive landscape
            4. Recent financial performance
            5. Management and leadership insights"""
        )
        
        # Creating a crew with the agents and tasks
        crew = Crew(
            agents=[stock_analyst, company_researcher],
            tasks=[stock_analysis_task, company_research_task]
        )
        
        # Running the crew and getting results
        result = crew.kickoff()
        st.sidebar.info("CrewAI Analysis Insights:")
        st.sidebar.write(result)
        
        return True
    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        return False

def get_stock_data(symbol):
    """
    Fetch stock data for NSE stocks
    Automatically appends .NS if not already present
    """
    try:
        # Ensure NSE suffix for Indian stocks
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        # Fetching stock information
        stock = yf.Ticker(symbol)
        
        # Getting historical data for price chart
        hist_data = stock.history(period="1y")
        
        # Getting stock info
        info = stock.info
        
        return info, hist_data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None, None

def create_price_chart(hist_data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index, open=hist_data['Open'],
        high=hist_data['High'], low=hist_data['Low'],
        close=hist_data['Close'], name='OHLC'
    ))
    fig.update_layout(
        title=f'{symbol} Price Movement',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=500
    )
    return fig

def main():
    st.title("Indian Stocks Analysis (CrewAI)")
    
    # Updating the help text to clarify NSE stocks
    stock_input = st.text_input("Enter Company Name", help="e.g., TCS, INFOSYS, RELIANCE (NSE Stocks)")
    
    if st.button("Analyze"):
        if not stock_input:
            st.error("Please enter a stock name")
            return
        
        # Use NSE stock mapping
        symbol = COMMON_STOCKS.get(stock_input.upper(), stock_input)
        
        # Initializing CrewAI agents
        initialize_agents(symbol)
        
        with st.spinner("Analyzing NSE Stock..."):
            info, hist = get_stock_data(symbol)
            
            if info and hist is not None:
                # Current price is already in INR for NSE stocks
                current_price_inr = f"₹{info.get('currentPrice', 'N/A'):.2f}" if info.get('currentPrice') != 'N/A' else 'N/A'
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='card'><div class='metric-value'>{current_price_inr}</div><div class='metric-label'>Current Price (INR)</div></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='card'><div class='metric-value'>{info.get('forwardPE', 'N/A')}</div><div class='metric-label'>Forward P/E</div></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='card'><div class='metric-value'>{info.get('recommendationKey', 'N/A').title()}</div><div class='metric-label'>Recommendation</div></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.plotly_chart(create_price_chart(hist, symbol))
                st.markdown("</div>", unsafe_allow_html=True)
                
                if 'longBusinessSummary' in info:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### Company Overview")
                    st.write(info['longBusinessSummary'])
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(f"Could not retrieve data for {symbol}. Please check the stock symbol.")

if __name__ == "__main__":
    main()