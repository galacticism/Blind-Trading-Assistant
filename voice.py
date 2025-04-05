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
from fredapi import Fred
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load environment variables
load_dotenv()

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize FRED API
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    print("Warning: FRED_API_KEY not found in .env file. Please add it.")
    FRED_API_KEY = "your_api_key_here"  # Placeholder for testing
fred = Fred(api_key=FRED_API_KEY)

# Initialize Whisper model for speech recognition
print("Loading Whisper model (this might take a moment)...")
whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
print("Whisper model loaded!")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Default recording time

# Economic indicators and their FRED series IDs
INDICATORS = {
    "unemployment rate": "UNRATE",
    "inflation": "CPIAUCSL",
    "gdp": "GDP",
    "federal funds rate": "FEDFUNDS",
    "consumer sentiment": "UMCSENT",
    "retail sales": "RSAFS",
    "housing starts": "HOUST",
    "industrial production": "INDPRO",
    "s&p 500": "SP500",
    "10-year treasury": "DGS10",
    "yield curve": {"10-year": "DGS10", "2-year": "DGS2"},
    "consumer price index": "CPIAUCSL",
    "personal income": "PI",
    "personal spending": "PCE",
}

def record_audio(seconds=RECORD_SECONDS):
    """Record audio from microphone and save to a temporary file"""
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Recording...")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to a temporary file
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
        temp_filename = record_audio()
        
        # Transcribe with Whisper
        print("Transcribing with Whisper...")
        result = whisper_model.transcribe(temp_filename)
        command = result["text"].lower().strip()
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        print(f"Command recognized: {command}")
        return command
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        speak("Sorry, I encountered an error while trying to understand your command.")
        return None

def speak(text):
    """Convert text to speech"""
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def extract_indicators(command):
    """Extract economic indicators mentioned in the command"""
    indicators = []
    tokens = word_tokenize(command)
    
    for indicator in INDICATORS.keys():
        if indicator in command:
            indicators.append(indicator)
    
    return indicators

def extract_timeframe(command):
    """Extract timeframe from command"""
    # Default timeframe: 1 year
    years = 1
    months = 0
    
    # Check for years
    year_match = re.search(r'(\d+)\s+year', command)
    if year_match:
        years = int(year_match.group(1))
    
    # Check for months
    month_match = re.search(r'(\d+)\s+month', command)
    if month_match:
        months = int(month_match.group(1))
    
    # Check for "last year", "last month"
    if "last year" in command and not year_match:
        years = 1
    if "last month" in command and not month_match:
        months = 1
        years = 0
    
    # Calculate observation date range
    if years > 0 or months > 0:
        return years, months
    else:
        return 1, 0  # Default to 1 year

def get_fred_data(series_id, years, months):
    """Get data from FRED API"""
    try:
        # Calculate observation period
        end_date = 'today'
        observation_period = years * 12 + months
        
        # Get data
        data = fred.get_series(series_id, end=end_date, observation_period=observation_period)
        return data
    except Exception as e:
        print(f"Error getting FRED data: {e}")
        return None

def plot_data(data, title):
    """Plot data and save to file"""
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    
    # Save file
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def process_command(command):
    """Process voice command and respond with FRED data"""
    if not command:
        return
    
    # Check for exit command
    if "exit" in command or "quit" in command:
        speak("Goodbye!")
        return False
    
    # Extract indicators
    indicators = extract_indicators(command)
    if not indicators:
        speak("I couldn't identify any economic indicators in your request. Please mention specific indicators like unemployment rate, inflation, or GDP.")
        return True
    
    # Extract timeframe
    years, months = extract_timeframe(command)
    timeframe_text = f"{years} year{'s' if years != 1 else ''}" if years > 0 else ""
    if months > 0:
        if timeframe_text:
            timeframe_text += f" and {months} month{'s' if months != 1 else ''}"
        else:
            timeframe_text = f"{months} month{'s' if months != 1 else ''}"
    
    for indicator in indicators:
        series_id = INDICATORS[indicator]
        
        # Special handling for yield curve
        if indicator == "yield curve":
            speak(f"Retrieving {indicator} data for the past {timeframe_text}...")
            try:
                ten_year = get_fred_data(series_id["10-year"], years, months)
                two_year = get_fred_data(series_id["2-year"], years, months)
                
                # Calculate spread
                spread = ten_year - two_year
                
                # Create DataFrame for plotting
                df = pd.DataFrame({
                    "10-Year Treasury": ten_year,
                    "2-Year Treasury": two_year,
                    "Spread (10Y-2Y)": spread
                })
                
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.plot(df["10-Year Treasury"], label="10-Year Treasury")
                plt.plot(df["2-Year Treasury"], label="2-Year Treasury")
                plt.title("Treasury Yields")
                plt.grid(True)
                plt.legend()
                
                plt.subplot(2, 1, 2)
                plt.plot(df["Spread (10Y-2Y)"], color="green")
                plt.axhline(y=0, color="red", linestyle="--")
                plt.title("Yield Curve Spread (10Y-2Y)")
                plt.grid(True)
                
                plt.tight_layout()
                filename = "yield_curve.png"
                plt.savefig(filename)
                plt.close()
                
                latest_10y = ten_year.iloc[-1] if not ten_year.empty else "N/A"
                latest_2y = two_year.iloc[-1] if not two_year.empty else "N/A"
                latest_spread = spread.iloc[-1] if not spread.empty else "N/A"
                
                speak(f"The current 10-year Treasury yield is {latest_10y:.2f}% and the 2-year yield is {latest_2y:.2f}%. "
                      f"The spread is {latest_spread:.2f}%. I've saved a chart as {filename}.")
                
            except Exception as e:
                speak(f"Sorry, I encountered an error retrieving yield curve data: {e}")
            
        else:
            speak(f"Retrieving {indicator} data for the past {timeframe_text}...")
            data = get_fred_data(series_id, years, months)
            
            if data is not None and not data.empty:
                # Plot data
                title = f"{indicator.title()} - Past {timeframe_text}"
                filename = plot_data(data, title)
                
                # Get latest value and percent change
                latest_value = data.iloc[-1]
                earliest_value = data.iloc[0]
                pct_change = ((latest_value - earliest_value) / earliest_value) * 100 if earliest_value != 0 else 0
                
                # Generate response
                response = f"The current {indicator} is {latest_value:.2f}. "
                if pct_change >= 0:
                    response += f"It has increased by {pct_change:.2f}% over the past {timeframe_text}."
                else:
                    response += f"It has decreased by {abs(pct_change):.2f}% over the past {timeframe_text}."
                    
                response += f" I've saved a chart as {filename}."
                speak(response)
            else:
                speak(f"Sorry, I couldn't retrieve data for {indicator}.")
    
    return True

def main():
    speak("Blind Trading Assistant activated. How can I help you with economic data today?")
    
    running = True
    while running:
        command = listen_for_command()
        running = process_command(command)

if __name__ == "__main__":
    main()
