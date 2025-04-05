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
from datetime import datetime
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize

# Load environment variables (not needed now but kept for future)
load_dotenv()

# Download NLTK resources
nltk.download('punkt', quiet=True)

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

# Load CSV files
def load_csv_files():
    """Load all economic indicator CSV files"""
    data = {}
    try:
        data["unemployment"] = pd.read_csv("Unemployment.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["rgdp"] = pd.read_csv("RGDP.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["industrial_production"] = pd.read_csv("Industrial Production.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["investment"] = pd.read_csv("GPDI.csv", parse_dates=["observation_date"], index_col="observation_date")
        data["consumption"] = pd.read_csv("Personal Consumption Expenditures.csv", parse_dates=["observation_date"], index_col="observation_date")
        
        print("CSV files loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None

# Economic indicators and their corresponding CSV file keys and display names
INDICATORS = {
    "unemployment": {
        "data_key": "unemployment", 
        "name": "Unemployment Rate", 
        "column": "LRUN64TTUSQ156S",
        "unit": "%",
        "description": "the percentage of the labor force that is jobless and actively seeking employment",
        "change_type": "percentage_point"  # Use percentage point change (subtract)
    },
    "unemployment rate": {
        "data_key": "unemployment", 
        "name": "Unemployment Rate", 
        "column": "LRUN64TTUSQ156S",
        "unit": "%",
        "description": "the percentage of the labor force that is jobless and actively seeking employment",
        "change_type": "percentage_point"  # Use percentage point change (subtract)
    },
    "gdp": {
        "data_key": "rgdp", 
        "name": "Real GDP", 
        "column": "GDPC1",
        "unit": "billion dollars",
        "description": "the inflation-adjusted value of all goods and services produced by the economy",
        "change_type": "absolute"  # Use absolute change (subtract)
    },
    "real gdp": {
        "data_key": "rgdp", 
        "name": "Real GDP", 
        "column": "GDPC1",
        "unit": "billion dollars",
        "description": "the inflation-adjusted value of all goods and services produced by the economy",
        "change_type": "absolute"  # Use absolute change (subtract)
    },
    "industrial production": {
        "data_key": "industrial_production", 
        "name": "Industrial Production", 
        "column": "INDPRO",
        "unit": "index (2017=100)",
        "description": "a measure of output from manufacturing, mining, electric and gas utilities",
        "change_type": "percentage"  # Use percentage change
    },
    "investment": {
        "data_key": "investment", 
        "name": "Gross Private Domestic Investment", 
        "column": "GPDI",
        "unit": "billion dollars",
        "description": "the measurement of physical investment used in GDP",
        "change_type": "absolute"  # Use absolute change (subtract)
    },
    "consumption": {
        "data_key": "consumption", 
        "name": "Personal Consumption Expenditures", 
        "column": "PCE",
        "unit": "billion dollars",
        "description": "a measure of consumer spending on goods and services",
        "change_type": "absolute"  # Use absolute change (subtract)
    },
    "personal consumption": {
        "data_key": "consumption", 
        "name": "Personal Consumption Expenditures", 
        "column": "PCE",
        "unit": "billion dollars",
        "description": "a measure of consumer spending on goods and services",
        "change_type": "absolute"  # Use absolute change (subtract)
    }
}

# Load data at startup
ECONOMIC_DATA = load_csv_files()

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
    
    for indicator in INDICATORS.keys():
        if indicator in command:
            indicators.append(indicator)
    
    return indicators

def extract_timeframe(command):
    """Extract timeframe from command"""
    # Default timeframe: 1 year
    years = 1
    months = 0
    
    # Check for "all" or "entire" dataset
    if "all" in command or "entire" in command or "last 10 years" in command:
        return 10, 0  # Return the entire dataset (approximately 10 years)
    
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
        months = 3  # Use last quarter since data is quarterly
        years = 0
    
    # Calculate observation date range
    if years > 0 or months > 0:
        return years, months
    else:
        return 1, 0  # Default to 1 year

def get_data_for_indicator(indicator, years, months):
    """Get data for an indicator over the specified timeframe"""
    if not ECONOMIC_DATA:
        print("Economic data not loaded")
        return None
    
    try:
        indicator_info = INDICATORS[indicator]
        data_key = indicator_info["data_key"]
        column = indicator_info["column"]
        
        # Get data from the loaded CSV
        df = ECONOMIC_DATA[data_key]
        
        # Sort by date to ensure proper order
        df = df.sort_index()
        
        # If we want all data, return all
        if years >= 10:
            return df[column]
        
        # Calculate the date range
        now = datetime.now()
        quarters = years * 4 + (months // 3)
        if quarters < 1:
            quarters = 1
            
        # Take the last N quarters of data
        data = df[column].iloc[-quarters:]
        
        return data
    except Exception as e:
        print(f"Error retrieving data for {indicator}: {e}")
        return None

def plot_data(data, title, y_label=None):
    """Plot data and save to file"""
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    if y_label:
        plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    
    # Save file
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
        # Get latest values
        unemployment = ECONOMIC_DATA["unemployment"]["LRUN64TTUSQ156S"].iloc[-1]
        rgdp = ECONOMIC_DATA["rgdp"]["GDPC1"].iloc[-1]
        
        # Calculate recent GDP growth (last quarter)
        gdp_growth_dollars = ECONOMIC_DATA["rgdp"]["GDPC1"].iloc[-1] - ECONOMIC_DATA["rgdp"]["GDPC1"].iloc[-2]
        
        # Add to summary
        summary += f"The current unemployment rate is {unemployment:.1f}%. "
        summary += f"Real GDP is at {rgdp:.1f} billion dollars, "
        summary += f"with {gdp_growth_dollars:.1f} billion dollars growth in the last quarter. "
        
        return summary
    except Exception as e:
        print(f"Error generating economic summary: {e}")
        return "I'm having trouble summarizing the current economic data."

def compare_to_historical(indicator, data):
    """Generate comparison to historical highs, lows, and averages"""
    # Get full dataset
    indicator_info = INDICATORS[indicator]
    data_key = indicator_info["data_key"]
    column = indicator_info["column"]
    full_data = ECONOMIC_DATA[data_key][column]
    
    current = data.iloc[-1]
    historical_max = full_data.max()
    historical_min = full_data.min()
    historical_avg = full_data.mean()
    
    percentile = (full_data < current).mean() * 100
    
    # Get 2019 pre-pandemic average for comparison
    pre_pandemic = full_data['2019'].mean()
    
    # Handle different types of changes
    if indicator_info["change_type"] == "percentage_point":
        vs_pre_pandemic = current - pre_pandemic
        vs_pre_text = f"{abs(vs_pre_pandemic):.1f} percentage points"
    elif indicator_info["change_type"] == "absolute":
        vs_pre_pandemic = current - pre_pandemic
        vs_pre_text = f"{abs(vs_pre_pandemic):.1f} {indicator_info['unit']}"
    else:  # percentage
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
    
    # Report the change compared to pre-pandemic
    if indicator_info["change_type"] == "percentage_point" or indicator_info["change_type"] == "absolute":
        if vs_pre_pandemic > 0:
            comparison += f"It is {vs_pre_text} higher than before the pandemic. "
        else:
            comparison += f"It is {vs_pre_text} lower than before the pandemic. "
    else:  # percentage
        if vs_pre_pandemic > 0:
            comparison += f"It is {vs_pre_text} higher than before the pandemic. "
        else:
            comparison += f"It is {vs_pre_text} lower than before the pandemic. "
    
    return comparison

def format_change(indicator, latest_value, earliest_value):
    """Format change between values according to indicator type"""
    indicator_info = INDICATORS[indicator]
    
    # Handle different types of changes
    if indicator_info["change_type"] == "percentage_point":
        # For percentage point indicators like unemployment rate
        change_value = latest_value - earliest_value
        if change_value >= 0:
            return f"It has increased by {change_value:.1f} percentage points"
        else:
            return f"It has decreased by {abs(change_value):.1f} percentage points"
    
    elif indicator_info["change_type"] == "absolute":
        # For absolute value indicators like GDP
        change_value = latest_value - earliest_value
        if change_value >= 0:
            return f"It has increased by {change_value:.1f} {indicator_info['unit']}"
        else:
            return f"It has decreased by {abs(change_value):.1f} {indicator_info['unit']}"
    
    else:  # "percentage" - default percentage change
        # For percentage change indicators like industrial production
        pct_change = ((latest_value - earliest_value) / earliest_value) * 100 if earliest_value != 0 else 0
        if pct_change >= 0:
            return f"It has increased by {pct_change:.1f}%"
        else:
            return f"It has decreased by {abs(pct_change):.1f}%"

def process_command(command):
    """Process voice command and respond with economic data"""
    if not command:
        return True
    
    # Check for exit command
    if "exit" in command or "quit" in command:
        speak("Goodbye!")
        return False
    
    # Check for help command
    if "help" in command or "what can you do" in command:
        help_text = "I can provide economic data on unemployment, GDP, industrial production, investment, and consumption. "
        help_text += "You can ask questions like: 'What is the current unemployment rate?', "
        help_text += "'How has GDP changed over the last 5 years?', or "
        help_text += "'Compare industrial production to historical data'. "
        speak(help_text)
        return True
    
    # Extract indicators
    indicators = extract_indicators(command)
    if not indicators:
        speak("I couldn't identify any economic indicators in your request. Please mention specific indicators like unemployment rate, GDP, industrial production, investment, or consumption.")
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
        indicator_info = INDICATORS[indicator]
        indicator_name = indicator_info["name"]
        
        speak(f"Retrieving {indicator_name} data for the past {timeframe_text}...")
        data = get_data_for_indicator(indicator, years, months)
        
        if data is not None and not data.empty:
            # Plot data
            title = f"{indicator_name} - Past {timeframe_text}"
            y_label = f"{indicator_name} ({indicator_info['unit']})"
            filename = plot_data(data, title, y_label)
            
            # Get latest value and format change
            latest_value = data.iloc[-1]
            earliest_value = data.iloc[0]
            change_text = format_change(indicator, latest_value, earliest_value)
            
            # Generate response
            response = f"The current {indicator_name} is {latest_value:.2f} {indicator_info['unit']}. "
            response += f"{change_text} over the past {timeframe_text}. "
            
            # Add historical comparison if requested
            if "compare" in command or "historical" in command or "history" in command or "past" in command:
                response += compare_to_historical(indicator, data)
                
            response += f"{indicator_info['description'].capitalize()}. "
            response += f"I've saved a chart as {filename}."
            speak(response)
        else:
            speak(f"Sorry, I couldn't retrieve data for {indicator_name}.")
    
    return True

def main():
    if not ECONOMIC_DATA:
        speak("Warning: I couldn't load the economic data files. Please check that the CSV files are in the current directory.")
        return
    
    # Provide a summary of current economic conditions on startup
    summary = get_current_summary()
    speak("Blind Trading Assistant activated. " + summary + " How can I help you today?")
    
    running = True
    while running:
        command = listen_for_command()
        running = process_command(command)

if __name__ == "__main__":
    main()
