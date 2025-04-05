import os
import re
import time
import whisper
import pyaudio
import wave
import tempfile
import numpy as np
import pyttsx3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
import audioop
from array import array
import yfinance as yf
import threading
import queue
from functools import lru_cache

load_dotenv()

nltk.download('punkt', quiet=True)

# Use 'tiny' model instead of 'base' for faster processing
print("Loading Whisper model (this might take a moment)...")
whisper_model = whisper.load_model("tiny")
print("Whisper model loaded!")

engine = pyttsx3.init()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1.5
MAX_RECORD_SECONDS = 30

# Queue for background processing tasks
task_queue = queue.Queue()

# Preload and cache CSV data
def load_csv_files():
    """Load all economic indicator CSV files"""
    data = {}
    try:
        data["unemployment"] = pd.read_csv("Unemployment.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["rgdp"] = pd.read_csv("RGDP.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["industrial_production"] = pd.read_csv("Industrial Production.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["investment"] = pd.read_csv("GPDI.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["consumption"] = pd.read_csv("Personal Consumption Expenditures.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["cpi"] = pd.read_csv("CPI.csv", parse_dates=["observation_date"], index_col="observation_date")
        
        # Sort data once at load time
        for key in data:
            data[key] = data[key].sort_index()
        
        print("CSV files loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None

# economic indicators if you guys know some more feel free to add, idk much
INDICATORS = {
    "unemployment": {
        "data_key": "unemployment", 
        "name": "Unemployment Rate", 
        "column": "LRUN64TTUSQ156S",
        "unit": "%",
        "description": "the percentage of the labor force that is jobless and actively seeking employment",
        "change_type": "percentage_point" 
    },
    "unemployment rate": {
        "data_key": "unemployment", 
        "name": "Unemployment Rate", 
        "column": "LRUN64TTUSQ156S",
        "unit": "%",
        "description": "the percentage of the labor force that is jobless and actively seeking employment",
        "change_type": "percentage_point"
    },
    "gdp": {
        "data_key": "rgdp", 
        "name": "Real GDP", 
        "column": "GDPC1",
        "unit": "billion dollars",
        "description": "the inflation-adjusted value of all goods and services produced by the economy",
        "change_type": "absolute"
    },
    "real gdp": {
        "data_key": "rgdp", 
        "name": "Real GDP", 
        "column": "GDPC1",
        "unit": "billion dollars",
        "description": "the inflation-adjusted value of all goods and services produced by the economy",
        "change_type": "absolute"
    },
    "industrial production": {
        "data_key": "industrial_production", 
        "name": "Industrial Production", 
        "column": "INDPRO",
        "unit": "index (2017=100)",
        "description": "a measure of output from manufacturing, mining, electric and gas utilities",
        "change_type": "percentage"
    },
    "investment": {
        "data_key": "investment", 
        "name": "Gross Private Domestic Investment", 
        "column": "GPDI",
        "unit": "billion dollars",
        "description": "the measurement of physical investment used in GDP",
        "change_type": "absolute"
    },
    "consumption": {
        "data_key": "consumption", 
        "name": "Personal Consumption Expenditures", 
        "column": "PCE",
        "unit": "billion dollars",
        "description": "a measure of consumer spending on goods and services",
        "change_type": "absolute"
    },
    "personal consumption": {
        "data_key": "consumption", 
        "name": "Personal Consumption Expenditures", 
        "column": "PCE",
        "unit": "billion dollars",
        "description": "a measure of consumer spending on goods and services",
        "change_type": "absolute"
    },
    "cpi": {
        "data_key": "cpi", 
        "name": "Consumer Price Index", 
        "column": "CPALTT01USQ657N",
        "unit": "%",
        "description": "a measure of the average change over time in the prices paid by consumers for a basket of goods and services",
        "change_type": "percentage"
    },
    "inflation": {
        "data_key": "cpi", 
        "name": "Consumer Price Index", 
        "column": "CPALTT01USQ657N",
        "unit": "%",
        "description": "a measure of the average change over time in the prices paid by consumers for a basket of goods and services",
        "change_type": "percentage"
    },
    "consumer price index": {
        "data_key": "cpi", 
        "name": "Consumer Price Index", 
        "column": "CPALTT01USQ657N",
        "unit": "%",
        "description": "a measure of the average change over time in the prices paid by consumers for a basket of goods and services",
        "change_type": "percentage"
    }
}

# Stock ticker mappings - common company names to ticker symbols
STOCK_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "disney": "DIS",
    "walmart": "WMT",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "mcdonalds": "MCD",
    "starbucks": "SBUX",
    "nike": "NKE",
    "ibm": "IBM",
    "intel": "INTC",
    "ford": "F",
    "general motors": "GM",
    "exxon": "XOM",
    "exxon mobil": "XOM",
    "chevron": "CVX",
    "goldman sachs": "GS",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "visa": "V",
    "mastercard": "MA",
    "verizon": "VZ",
    "att": "T",
    "at&t": "T",
    "costco": "COST",
    "home depot": "HD",
    "target": "TGT",
    "sp500": "^GSPC",
    "s&p 500": "^GSPC",
    "s&p": "^GSPC",
    "dow": "^DJI",
    "dow jones": "^DJI",
    "nasdaq": "^IXIC",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD"
}

# Cache of stock data to avoid repeated API calls
STOCK_CACHE = {}

ECONOMIC_DATA = load_csv_files()

def is_silent(snd_data, threshold=SILENCE_THRESHOLD):
    """Returns True if the sound data is below the silence threshold"""
    # RMS to determine silence. Could use other method, but I think this is accurate enough and way simpler
    rms = audioop.rms(snd_data, 2)
    return rms < threshold

def normalize(snd_data):
    """Average the volume out"""
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r

def record_audio_with_silence_detection():
    """Record audio from microphone until silence is detected"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Listening... (speak when ready, I'll process when you stop talking)")
    
    frames = []
    silent_frames = 0
    max_frames = int(RATE / CHUNK * MAX_RECORD_SECONDS)
    
    # wait to process until you hear speech, don't want to repeatedly process silence for no reason
    silent_count = 0
    while True:
        data = stream.read(CHUNK)
        if not is_silent(data):
            break
        silent_count += 1
        # If silence for too long, prompt user with something to make sure they're not afk
        if silent_count > int(RATE / CHUNK * 5):
            print("Waiting for speech...")
            silent_count = 0
    
    print("Speech detected, recording...")
    
    # silence for 1.5 seconds after speech means we process, because the user is prob done speaking
    for i in range(max_frames):
        data = stream.read(CHUNK)
        frames.append(data)
        
        if is_silent(data):
            silent_frames += 1
            if silent_frames >= int(SILENCE_DURATION * RATE / CHUNK):
                break
        else:
            silent_frames = 0
            
    print("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_filename = temp_audio.name
        
    wf = wave.open(temp_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return temp_filename

def listen_for_command():
    """Listen for voice command using microphone and transcribe with Whisper"""
    try:
        temp_filename = record_audio_with_silence_detection()
        
        # Quick acknowledgment to let user know system is working
        speak("Processing...", wait=False)
        
        # turn into text
        print("Transcribing with Whisper...")
        start_time = time.time()
        result = whisper_model.transcribe(temp_filename)
        command = result["text"].lower().strip()
        print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        
        os.unlink(temp_filename)
        
        print(f"Command recognized: {command}")
        return command
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        speak("Sorry, I encountered an error while trying to understand your command.")
        return None

def speak(text, wait=True):
    """Convert text to speech"""
    print(f"Assistant: {text}")
    engine.say(text)
    if wait:
        engine.runAndWait()
    else:
        # Start a thread to handle the speech in the background
        # and return immediately to the user
        threading.Thread(target=engine.runAndWait).start()

def extract_indicators(command):
    """Extract economic indicators mentioned in the command"""
    indicators = []
    matched_data_keys = set()
    
    sorted_keys = sorted(INDICATORS.keys(), key=len, reverse=True)
    
    for indicator in sorted_keys:
        if indicator in command:
            data_key = INDICATORS[indicator]["data_key"]
            if data_key not in matched_data_keys:
                indicators.append(indicator)
                matched_data_keys.add(data_key)
    
    return indicators

def extract_ticker(command):
    """Extract stock ticker from command"""
    # First check if the command contains any company names from our mapping
    command_lower = command.lower()
    for company, ticker in STOCK_TICKERS.items():
        if company in command_lower:
            return ticker, company
    
    # If not found, check for explicit ticker mentions (uppercase 1-5 letter sequences)
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', command)
    if ticker_match:
        ticker = ticker_match.group(1)
        return ticker, ticker
    
    return None, None

def extract_timeframe(command):
    """Extract timeframe from command"""
    # Default timeframe: 1 year
    years = 1
    months = 0
    
    if "all" in command or "entire" in command or "last 10 years" in command:
        return 10, 0
    
    # Check for years
    year_match = re.search(r'(\d+)\s+year', command)
    if year_match:
        years = int(year_match.group(1))
    
    # Check for months
    month_match = re.search(r'(\d+)\s+month', command)
    if month_match:
        months = int(month_match.group(1))
    
    # Check for "last"
    if "last year" in command and not year_match:
        years = 1
    if "last month" in command and not month_match:
        months = 3  # quarterly data so 3 months instead of 1
        years = 0
    
    if years == 0 and months == 0:
        years = 1  # can't get data for less than a certain timeframe
        
    return years, months

def get_data_for_indicator(indicator, years, months):
    """Get data for an indicator over the specified timeframe"""
    if not ECONOMIC_DATA:
        print("Economic data not loaded")
        return None
    
    try:
        indicator_info = INDICATORS[indicator]
        data_key = indicator_info["data_key"]
        column = indicator_info["column"]
        
        df = ECONOMIC_DATA[data_key]
        
        if years >= 10:
            return df[column]
        
        now = datetime.now()
        quarters = years * 4 + (months // 3)
        if quarters < 1:
            quarters = 1
            
        data = df[column].iloc[-quarters:]
        
        return data
    except Exception as e:
        print(f"Error retrieving data for {indicator}: {e}")
        return None

# plot graph of economic data for user's request so they can see, for non-blind people
def plot_data(data, title, y_label=None):
    """Plot data and save to file"""
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    if y_label:
        plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def get_current_summary():
    """Generate a summary of current economic conditions based on the latest data"""
    if not ECONOMIC_DATA:
        return "Sorry, economic data is not available."
    
    summary = "Here's a summary of current economic conditions: "
    
    try:
        unemployment = ECONOMIC_DATA["unemployment"]["LRUN64TTUSQ156S"].iloc[-1]
        rgdp = ECONOMIC_DATA["rgdp"]["GDPC1"].iloc[-1]
        
        # CPI
        cpi_quarterly = ECONOMIC_DATA["cpi"]["CPALTT01USQ657N"]
        
        # annual inflation from 4 quarters of CPI
        annual_inflation = cpi_quarterly.iloc[-4:].sum()
        
        # RGDP growth quarterly
        gdp_growth_dollars = ECONOMIC_DATA["rgdp"]["GDPC1"].iloc[-1] - ECONOMIC_DATA["rgdp"]["GDPC1"].iloc[-2]
        
        summary += f"The current unemployment rate is {unemployment:.1f}%. "
        summary += f"Annual inflation is running at {annual_inflation:.1f}%. "
        summary += f"Real GDP is at {rgdp:.1f} billion dollars, "
        summary += f"with {gdp_growth_dollars:.1f} billion dollars growth in the last quarter. "
        
        return summary
    except Exception as e:
        print(f"Error generating economic summary: {e}")
        return "I'm having trouble summarizing the current economic data."

def compare_to_historical(indicator, data, timeframe_text):
    """Generate comparison to historical highs, lows, and averages"""
    indicator_info = INDICATORS[indicator]
    data_key = indicator_info["data_key"]
    column = indicator_info["column"]
    full_data = ECONOMIC_DATA[data_key][column]
    
    current = data.iloc[-1]
    historical_max = full_data.max()
    historical_min = full_data.min()
    historical_avg = full_data.mean()
    
    percentile = (full_data < current).mean() * 100
    
    # pre-pandemic avgs for comparison
    pre_pandemic = full_data['2019'].mean()
    
    # Handle different types of changes
    if indicator_info["change_type"] == "percentage_point":
        vs_pre_pandemic = current - pre_pandemic
        vs_pre_text = f"{abs(vs_pre_pandemic):.1f} percentage points"
    elif indicator_info["change_type"] == "absolute":
        vs_pre_pandemic = current - pre_pandemic
        vs_pre_text = f"{abs(vs_pre_pandemic):.1f} {indicator_info['unit']}"
    else:
        vs_pre_pandemic = ((current / pre_pandemic) - 1) * 100
        vs_pre_text = f"{abs(vs_pre_pandemic):.1f}%"
    
    comparison = f"The current value of {current:.2f} {indicator_info['unit']} "
    
    if abs(current - historical_max) < 0.01 * historical_max:
        comparison += "is near the historical high "
    elif abs(current - historical_min) < 0.01 * historical_max:
        comparison += "is near the historical low "
    else:
        comparison += f"is at the {percentile:.0f}th percentile "
    
    comparison += f"compared to the last 10 years. "
    comparison += f"The historical average is {historical_avg:.2f} {indicator_info['unit']}. "
    
    # change compared to pre-pandemic
    if indicator_info["change_type"] == "percentage_point" or indicator_info["change_type"] == "absolute":
        if vs_pre_pandemic > 0:
            comparison += f"It is {vs_pre_text} higher than before the pandemic. "
        else:
            comparison += f"It is {vs_pre_text} lower than before the pandemic. "
    else:
        if vs_pre_pandemic > 0:
            comparison += f"It is {vs_pre_text} higher than before the pandemic. "
        else:
            comparison += f"It is {vs_pre_text} lower than before the pandemic. "
    
    return comparison

def format_change(indicator, latest_value, earliest_value):
    """Format change between values according to indicator type"""
    indicator_info = INDICATORS[indicator]
    
    if indicator_info["change_type"] == "percentage_point":
        # percentage point indicators
        change_value = latest_value - earliest_value
        if change_value >= 0:
            return f"It has increased by {change_value:.1f} percentage points"
        else:
            return f"It has decreased by {abs(change_value):.1f} percentage points"
    
    elif indicator_info["change_type"] == "absolute":
        # absolute indicators
        change_value = latest_value - earliest_value
        if change_value >= 0:
            return f"It has increased by {change_value:.1f} {indicator_info['unit']}"
        else:
            return f"It has decreased by {abs(change_value):.1f} {indicator_info['unit']}"
    
    else: 
        pct_change = ((latest_value - earliest_value) / earliest_value) * 100 if earliest_value != 0 else 0
        if pct_change >= 0:
            return f"It has increased by {pct_change:.1f}%"
        else:
            return f"It has decreased by {abs(pct_change):.1f}%"

def calculate_annual_inflation(data, periods=4):
    """Calculate annual inflation rate from quarterly CPI data"""
    if len(data) < periods:
        return data.sum()
    
    return data.iloc[-periods:].sum()

@lru_cache(maxsize=32)
def get_stock_data(ticker):
    """Get stock data for a specific ticker with caching"""
    # Check cache first (for current session)
    if ticker in STOCK_CACHE:
        # Only use cached data if it's less than 10 minutes old
        cache_time, data = STOCK_CACHE[ticker]
        if datetime.now() - cache_time < timedelta(minutes=10):
            return data, None
    
    try:
        # Get stock info
        start_time = time.time()
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Get historical data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        hist = ticker_obj.history(start=start_date, end=end_date)
        print(f"Stock data retrieved in {time.time() - start_time:.2f} seconds")
        
        if hist.empty:
            return None, "No data available for this ticker."
        
        # Calculate performance metrics
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        day_change = ((current_price / prev_close) - 1) * 100
        
        month_start = hist['Close'].iloc[0]
        month_change = ((current_price / month_start) - 1) * 100
        
        # Get company name and sector
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        
        # Create a summary
        summary = {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'current_price': current_price,
            'day_change': day_change,
            'month_change': month_change,
            'history': hist
        }
        
        # Cache the result
        STOCK_CACHE[ticker] = (datetime.now(), summary)
        
        return summary, None
    except Exception as e:
        print(f"Error retrieving stock data: {e}")
        return None, f"Error retrieving stock data: {e}"

def plot_stock_data(data, ticker):
    """Plot stock price history"""
    hist = data['history']
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist.index, hist['Close'])
    plt.title(f"{data['company_name']} ({ticker}) - Last 30 Days")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"{ticker.lower()}_stock_price.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def generate_stock_response(data):
    """Generate a spoken response about stock performance"""
    ticker = data['ticker']
    company_name = data['company_name']
    current_price = data['current_price']
    day_change = data['day_change']
    month_change = data['month_change']
    sector = data['sector']
    
    response = f"{company_name} with ticker symbol {ticker} "
    
    if sector != 'N/A':
        response += f"in the {sector} sector "
    
    response += f"is currently trading at ${current_price:.2f}. "
    
    # Add day change
    if day_change > 0:
        response += f"It's up {abs(day_change):.1f}% today. "
    elif day_change < 0:
        response += f"It's down {abs(day_change):.1f}% today. "
    else:
        response += f"Its price is unchanged today. "
    
    # Add month change
    if month_change > 0:
        response += f"Over the past month, it has gained {abs(month_change):.1f}%. "
    elif month_change < 0:
        response += f"Over the past month, it has lost {abs(month_change):.1f}%. "
    else:
        response += f"Over the past month, its price has remained stable. "
    
    return response

def background_plot_task(func, *args, **kwargs):
    """Run plotting tasks in background to avoid blocking"""
    result = func(*args, **kwargs)
    task_queue.put(result)
    return result

def process_command(command):
    """Process voice command and respond with economic data"""
    if not command:
        return True
    
    # exit command
    if "exit" in command or "quit" in command:
        speak("Goodbye!")
        return False
    
    # functionality to help user if needed
    if "help" in command or "what can you do" in command:
        help_text = "I can provide economic data on unemployment, GDP, industrial production, investment, and consumption. "
        help_text += "You can ask questions like: 'What is the current unemployment rate?', "
        help_text += "'How has GDP changed over the last 5 years?', or "
        help_text += "'Compare industrial production to historical data'. "
        help_text += "I can also give you stock information. Just ask about a company like 'How is Apple doing?' or 'Tell me about Tesla stock.'"
        speak(help_text)
        return True
    
    # Check for stock ticker request
    if "stock" in command or "price" in command or "ticker" in command or "trading" in command or \
       "how is" in command or "tell me about" in command or "what about" in command:
        
        ticker, company_name = extract_ticker(command)
        
        if ticker:
            speak(f"Looking up stock information for {company_name}...", wait=False)
            
            # Use threading to fetch stock data in background
            def stock_processing():
                stock_data, error = get_stock_data(ticker)
                
                if error:
                    speak(f"Sorry, I couldn't retrieve stock data for {company_name}. {error}")
                    return
                
                # Generate the plot in background thread
                filename = plot_stock_data(stock_data, ticker)
                
                # Construct and speak the response
                response = generate_stock_response(stock_data)
                response += f"I've saved a chart of the stock's performance as {filename}."
                speak(response)
            
            # Start processing in background thread and return immediately
            threading.Thread(target=stock_processing).start()
            return True
    
    # if we can't find any indicators, tell user
    indicators = extract_indicators(command)
    if not indicators:
        # Check if it might be a stock request without standard keywords
        ticker, company_name = extract_ticker(command)
        if ticker:
            speak(f"I'll interpret that as a request for stock information about {company_name}...", wait=False)
            
            # Use threading for background processing
            def process_stock_request():
                stock_data, error = get_stock_data(ticker)
                
                if error:
                    speak(f"Sorry, I couldn't retrieve stock data for {company_name}. {error}")
                    return
                
                filename = plot_stock_data(stock_data, ticker)
                response = generate_stock_response(stock_data)
                response += f"I've saved a chart of the stock's performance as {filename}."
                speak(response)
            
            threading.Thread(target=process_stock_request).start()
            return True
        else:
            speak("I couldn't identify any economic indicators or stock tickers in your request. Please mention specific indicators like unemployment rate, GDP, or company names like Apple or Microsoft.")
            return True
    
    # timeframe
    years, months = extract_timeframe(command)
    timeframe_text = f"{years} year{'s' if years != 1 else ''}" if years > 0 else ""
    if months > 0:
        if timeframe_text:
            timeframe_text += f" and {months} month{'s' if months != 1 else ''}"
        else:
            timeframe_text = f"{months} month{'s' if months != 1 else ''}"
    
    for indicator in indicators:
        indicator_info = INDICATORS[indicator]
        indicator_name = indicator_info["name"]
        
        speak(f"Processing {indicator_name} data...", wait=False)
        
        # Define processing function for background thread
        def process_indicator_data():
            data = get_data_for_indicator(indicator, years, months)
            
            if data is not None and not data.empty:
                title = f"{indicator_name} - Past {timeframe_text}"
                y_label = f"{indicator_name} ({indicator_info['unit']})"
                filename = plot_data(data, title, y_label)
                
                latest_value = data.iloc[-1]
                earliest_value = data.iloc[0]
                
                if indicator in ["cpi", "inflation", "consumer price index"]:
                    if len(data) >= 8: 
                        annual_inflation_current = calculate_annual_inflation(data.iloc[-4:])
                        
                        if years > 1:
                            annual_inflation_start = calculate_annual_inflation(data.iloc[:4])
                            change_text = format_change(indicator, annual_inflation_current, annual_inflation_start)
                            response = f"The current annual inflation rate is {annual_inflation_current:.1f}%. "
                            response += f"{change_text} compared to {years} years ago. "
                        else:
                            annual_inflation_previous = calculate_annual_inflation(data.iloc[-8:-4])
                            change_text = format_change(indicator, annual_inflation_current, annual_inflation_previous)
                            response = f"The current annual inflation rate is {annual_inflation_current:.1f}%. "
                            response += f"{change_text} compared to the previous year. "
                    else:
                        change_text = format_change(indicator, latest_value, earliest_value)
                        response = f"The current quarterly inflation rate is {latest_value:.2f}%. "
                        response += f"{change_text} over the past {timeframe_text}. "
                else:
                    change_text = format_change(indicator, latest_value, earliest_value)
                    response = f"The current {indicator_name} is {latest_value:.2f} {indicator_info['unit']}. "
                    response += f"{change_text} over the past {timeframe_text}. "
                
                if "compare" in command or "historical" in command or "history" in command or "past" in command:
                    response += compare_to_historical(indicator, data, timeframe_text)
                    
                response += f"{indicator_info['description'].capitalize()}. "
                response += f"I've saved a chart as {filename}."
                speak(response)
            else:
                speak(f"Sorry, I couldn't retrieve data for {indicator_name}.")
        
        # Process in background thread
        threading.Thread(target=process_indicator_data).start()
    
    return True

def main():
    if not ECONOMIC_DATA:
        speak("Warning: I couldn't load the economic data files. Please check that the CSV files are in the current directory.")
        return
    
    # startup summary of current economic conditions
    summary = get_current_summary()
    speak("Blind Trading Assistant activated. " + summary + " How can I help you today?")
    
    running = True
    while running:
        command = listen_for_command()
        running = process_command(command)

if __name__ == "__main__":
    main()
