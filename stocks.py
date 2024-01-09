# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:57:15 2022

@author: ferdo
"""

import pandas as pd
from yahoo_fin import stock_info as si
from yahoo_fin import news
import streamlit as st
from stocksymbol import StockSymbol
from datetime import datetime as dt, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
import nasdaqdatalink
import math
import numpy as np
from PIL import Image
#from pathlib import Path
#import os


api_key = 'db609a4e-d7d2-43ab-87bf-21961d2d7428'
ss = StockSymbol(api_key)

# Make web app view wide
st.set_page_config(layout="wide")

# Create containers for the web app layout
header = st.container()
base_info = st.container()
valuation = st.container()
comparison = st.container()

# RGB strings for color
red = 'rgb(238,42,42)'
green = 'rgb(118,180,76)'
orange = 'rgb(242,186,84)'

@st.cache_data  # Cache data load for faster app performance

# Function to get all tickers and markets
def ticker_list():
    symbol = pd.DataFrame()  # Empty dataframe for loop
    
    market_list = pd.DataFrame(ss.market_list)  # Get all market list
    
    for country in market_list[market_list['abbreviation'].isin(['uk','us','au'])]['abbreviation']:
        if not symbol.empty:
            region = pd.DataFrame(pd.DataFrame(ss.get_symbol_list(country)))
            symbol=pd.concat([symbol, region])
        else:
            symbol=pd.DataFrame(pd.DataFrame(ss.get_symbol_list(country)))
    
    return symbol

@st.cache_data # Cache data load for faster app performance
def bond_yield():
    # AAA US Corporate bond yield
    bond = nasdaqdatalink.get('ML/AAAEY')
    return bond

@st.cache_data
def shares_issued(ticker):
    #Scrape Shares Issued from Yahoo Finance balance Sheet page
    url = "https://finance.yahoo.com/quote/" + ticker + "/balance-sheet?p=" + ticker
    html = requests.get(url=url, headers = {'User-agent': 'Mozilla/5.0'}).text
    
    
    json_str = html.split('root.App.main =')[1].split(
                '(this)')[0].split(';\n}')[0].strip()
    data = json.loads(json_str)['context']['dispatcher']['stores'][
                    'QuoteTimeSeriesStore']['timeSeries']['annualShareIssued']
    # return data
    new_data = json.dumps(data).replace('{}', 'null')
    new_data = re.sub(r'\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}', r'\1', new_data)
    
    json_info = json.loads(new_data)
    
    df = pd.DataFrame(json_info)
    
    return df    

@st.cache_data
def yahoo_stats(stock):
    stats = si.get_stats(stock)
    val = si.get_stats_valuation(stock)
    return stats,val
#nws = news.get_yf_rss('nflx')

@st.cache_data
def yahoo_news(stock):
    nws = news.get_yf_rss(stock)
    return nws

@st.cache_data
def comp_prices(stocks):
    
    df = pd.DataFrame()
    start_date = dt.today()-timedelta(days=365)
    for tkr in stocks:
        
        df_tmp = si.get_data(ticker=tkr,start_date = start_date)   # Get price data for individual ticker
        df_tmp['change_pct'] = (df_tmp.adjclose.div(
            df_tmp[df_tmp.index==df_tmp.first_valid_index()]['adjclose'].values[0]))-1
        
        # Check if main dataframe exists and append df_tmp to df
        if not df.empty:
            df=df.append(df_tmp)
        else:
            df=df_tmp        
        
        df_tmp = None
            
    return df






# Web app elements code starts here

# Defining header container components
with header:
    # Adding logo to the page
    
    logo = Image.open("./ferdous.JPG")
    st.image(logo, use_column_width='always')
    
    # Page title and other text within the header container
    st.title('Stock Information and Comparison')
    st.markdown('This page uses data from Yahoo Finance to provide base stock information and valuation.')
    st.header('Disclaimer')
    st.markdown('This tool is not to be used as stock suggestion. This is built of education and entertainment purposes only. CONSULT A PROFESSIONAL FOR ADVICE ON ANY INVESTMENT.')
                
with base_info:
    
    # Create list of Stock Symbols
    df_symbol = ticker_list()
    
    # Create distinct list of symbols to be used in the user select options
    list_symbol = df_symbol['symbol'].unique().tolist()
    
    # Get user to select the stock they want to investigate
    main_stock = st.selectbox('Select one (1) ticker to analyse: ', list_symbol, key = 'select_symbol' ,)

    # Columns for data visuals
    base_col_1,base_col_2,base_col_3,base_col_4 = st.columns(4)
    
    
    # Use yahoo finance to get data for the main stock
    try:
        stock_stats,val_stats = yahoo_stats(main_stock)
    except:
        st.warning(f'Data for {main_stock} is currently unavailable.')
        st.stop()
    
    val_stats.columns=['Field','Value']
    
    
    # Extract EPS value from stock_stats DataFrame
    eps_value = float(stock_stats[stock_stats['Attribute']=='Diluted EPS (ttm)']['Value'].values[0])
    
    # If EPS value is missing ask user to enter value
    if math.isnan(eps_value):
        eps_value=st.number_input('Enter EPS value: ')
      
    # Get expected 5 year growth rate
    try:
        analyst = si.get_analysts_info(main_stock)['Growth Estimates']
        growth = analyst[analyst['Growth Estimates']=='Next 5 Years (per annum)'][main_stock].values[0]
        
    except:
        growth='0'

    current_price = si.get_live_price(main_stock)
    base_col_1.metric("Current Price", round(current_price,2))        
    base_col_2.metric("Market Cap",val_stats[val_stats['Field']=='Market Cap (intraday)']['Value'].values[0])
    base_col_3.metric("EV/EBITDA",val_stats[val_stats['Field']=='Enterprise Value/EBITDA']['Value'].values[0])
    base_col_4.metric("Forward P/E",val_stats[val_stats['Field']=='Forward P/E']['Value'].values[0] )
    
    base_col_1.metric("EPS", eps_value)
    base_col_2.metric("Book Value per share",stock_stats[stock_stats['Attribute']=='Book Value Per Share (mrq)']['Value'].values[0] )
    base_col_3.metric("Cash per share",stock_stats[stock_stats['Attribute']=='Total Cash Per Share (mrq)']['Value'].values[0] )
    base_col_4.metric("FCF (levered)",stock_stats[stock_stats['Attribute']=='Levered Free Cash Flow (ttm)']['Value'].values[0])

    
    
    # Create Gauge charts for certain metrics
    current_ratio = float(stock_stats[stock_stats['Attribute']=='Current Ratio (mrq)']['Value'].values[0])
    current_ratio_chart = go.Figure(
                            go.Indicator(
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            value = current_ratio,
                            #number = {'suffix': '%'},
                            mode = "gauge+number",
                            title = {'text': "Current Ratio"},
                            gauge = {'axis': {'range': [None, 6.0]},
                                     'bar': {'color': "darkblue"},
                                     'steps' : [
                                         {'range': [0, 1.0], 'color': red},
                                         {'range': [1.0, 2.5], 'color': orange},
                                         {'range': [2.5, 15.0], 'color': green}],
                                     }
                            )
                        )
    current_ratio_chart.update_layout(
		width=200,
		height=250,
		margin=dict(l=1,r=1,b=1,t=1),
		#font=dict(color='#383635', size=15)
        )
    base_col_1.write(current_ratio_chart)
    
    # STock Performance gauge
    stock_perf = float(str(stock_stats[stock_stats['Attribute']=='52-Week Change 3']['Value'].values[0]).replace('%',''))
    stock_perf_chart = go.Figure(
                            go.Indicator(
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            value = stock_perf,
                            number = {'suffix': '%',
                                      'font':{'size':20}},
                            mode = "gauge+number",
                            title = {'text': f"{main_stock} (1 yr)",
                                     'font':{'size':15}},
                            gauge = {'axis': {'range': [-100, 100]},
                                     'bar': {'color': "darkblue"},
                                     'steps' : [
                                         {'range': [-100, 0.0], 'color': red},
                                         {'range': [0.0, 100.0], 'color': green}],
                                     }
                            )
                        )
    stock_perf_chart.update_layout(
		width=200,
		height=250,
		margin=dict(l=1,r=1,b=1,t=1),
		#font=dict(color='#383635', size=15)
        )   
       
    base_col_2.write(stock_perf_chart)
    
    
    
    # S&P500 last year Performance gauge
    sp_perf = float(str(stock_stats[stock_stats['Attribute']=='S&P500 52-Week Change 3']['Value'].values[0]).replace('%',''))
    sp_perf_chart = go.Figure(
                            go.Indicator(
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            value = sp_perf,
                            number = {'suffix': '%',
                                      'font':{'size':20}},
                            mode = "gauge+number",
                            title = {'text': "S&P500 (1 yr)",
                                     'font':{'size':15}},
                            gauge = {'axis': {'range': [-100, 100]},
                                     'bar': {'color': "darkblue"},
                                     'steps' : [
                                         {'range': [-100, 0.0], 'color': red},
                                         {'range': [0.0, 100.0], 'color': green}],
                                     }
                            )
                        )
    sp_perf_chart.update_layout(
		width=200,
		height=250,
		margin=dict(l=1,r=1,b=1,t=1),
		#font=dict(color='#383635', size=15)
        )   
    
    base_col_3.write(sp_perf_chart)


    # Return on Equity gauge
    roe = float(str(stock_stats[stock_stats['Attribute']=='Return on Equity (ttm)']['Value'].values[0]).replace('%',''))
    roe_chart = go.Figure(
                            go.Indicator(
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            value = roe,
                            number = {'suffix': '%',
                                      'font':{'size':20}},
                            mode = "gauge+number",
                            title = {'text': "ROE",
                                     'font':{'size':20}},
                            gauge = {'axis': {'range': [0, 50]},
                                     'bar': {'color': "darkblue"},
                                     'steps' : [
                                         {'range': [0.0, 7.5], 'color': red},
                                         {'range': [7.5,15.0], 'color': orange},
                                         {'range': [15.0, 50.0], 'color': green}],
                                     }
                            )
                        )
    roe_chart.update_layout(
		width=200,
		height=250,
		margin=dict(l=1,r=1,b=1,t=1),
		#font=dict(color='#383635', size=15)
        )   
    
    base_col_4.write(roe_chart)

    
with valuation:

    
    # Get AAA corporate bond yield from NASDAQ datalink
    yield_data = bond_yield()
    
    # Calculate average bond yld for the last 5 years
    avg_yld = yield_data[yield_data.index >= (dt.today()-timedelta(days=1826))]['BAMLC0A1CAAAEY'].mean() # Filtering data to last 5 years only
    
    # Get current yield
    max_date = yield_data.index.max()
    current_yield = yield_data[yield_data.index==max_date]['BAMLC0A1CAAAEY'].values[0]
    #P/E ratio of no growth company
    pe = 8.5
    
    

    
    # Allow user to choose a different Growth rate
    g = st.slider(label='Expected Growth Rate',min_value=0.0,max_value=30.0,value=0.0)
    
    # If user does not select a growth value revert to value from yahoo
    if g==0:
        g=float(str(growth).replace('%',''))
    
    # If user growth value and yahoo growth value is 0, ask user to input value
    if g==0 or math.isnan(g):
        st.warning('Expected growth data is unavailable on Yahoo Finance. Please use the slider above to select a growth value')
    else:
        if eps_value<=0:
            st.warning('Earnings data for the selected company is not available')
        else:
            V=(eps_value*(8.5+(2*g))*avg_yld)/current_yield
            if V>0:                
                st.info(f' Ben Graham Valuation: ${round(V,2)}\n\nAAA Corporate bond yield (5yr avg.): {round(avg_yld,2)}\
                        \n AAA Corporate bond yield (current): {round(current_yield,2)}\
                        \n Expected Growth Rate: {round(g,2)}%\
                        \n Current EPS: {eps_value}', icon="ℹ️")
            else:
                st.warning(f'Unable to calculate stock value\
                           \n Expected Growth Rate: {round(g,2)}%\
                           \n Current EPS: {eps_value}')
        
    #st.write(g, current_yield)        
    
    
    st.subheader(f'{main_stock} Prices')
    start_date = dt.today()-timedelta(days=365)
    price_data = si.get_data(main_stock, start_date = start_date)
    
    candle = go.Figure()
    candle.add_trace(go.Candlestick(name = f'{main_stock}',
                                    x = price_data.index, 
                                     close = price_data['adjclose'],
                                     open = price_data['open'], 
                                     high =price_data['high'], 
                                     low = price_data['low'] 
                                     )
                      )
    candle.update_layout(
		width=850*1.5,
		height=600,
		margin=dict(l=1,r=1,b=1,t=1),
		#font=dict(color='#383635', size=15)
        )   
       
    st.write(candle, use_container_width=True)
    
with comparison:

    # Get list of tickers to compare with the main stock
    comp_stock = st.multiselect(f'Select stock to compare to {main_stock}',df_symbol, key='compare_stock', default=['MSFT','INTC'])
    comp_stock.append(main_stock)    
    
    all_prices = comp_prices(comp_stock)
    all_prices.index.rename('Dates')
    
    comp_chart = px.line(all_prices,y='change_pct',color='ticker')
    # Edit the layout
    
    comp_chart.update_layout(title='Performance Comparison',
                    xaxis_title='Dates',
                    yaxis_title='Performance',
                    width=850*1.5,
               		height=600,
               		margin=dict(l=1,r=1,b=1,t=1))
    st.write(comp_chart, use_container_width = True)
    
with st.sidebar:
    st.header('News Articles')
    nws = yahoo_news(main_stock)
    for i in nws:
        st.markdown(i['published'][:16])
        #st.markdown(i['title'])
        with st.expander(i['title']):
            st.write(i['summary']+'\t'+'Use link for detail: '+i['link']+'\n')
        
        
        #st.markdown(i['summary']+'\t'+'Use link for detail: '+i['link']+'\n')
    
    
    
                    
    
