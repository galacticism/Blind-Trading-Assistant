# Blind Trading Assistant

A voice-activated trading assistant for blind users that provides economic data analysis through voice commands. This application uses OpenAI's Whisper for speech recognition and local CSV data files to provide information about key economic indicators.

## Features

- Voice command recognition using OpenAI Whisper
- Economic data analysis from local CSV files
- Automatic data visualization with saved charts
- Historical comparisons of economic indicators
- Voice responses optimized for blind users

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Note: You might need to install additional system dependencies for PyAudio:
   - On macOS: `brew install portaudio`
   - On Ubuntu/Debian: `sudo apt-get install python3-pyaudio portaudio19-dev`
   - On Windows: PyAudio wheel should install automatically from pip

3. Make sure your CSV files are in the same directory as the script:
   - Unemployment.csv
   - RGDP.csv (Real GDP)
   - Industrial Production.csv
   - GPDI.csv (Gross Private Domestic Investment)
   - Personal Consumption Expenditures.csv

## Usage

Run the assistant:
```
python voice.py
```

The assistant will:
1. Load the Whisper speech recognition model (this may take a moment on first run)
2. Load all CSV data files 
3. Provide a summary of current economic conditions
4. Listen for your voice commands (recording for 5 seconds by default)
5. Process the command and respond verbally
6. Generate and save visualizations when appropriate

Example commands:
- "What is the current unemployment rate?"
- "How has GDP changed over the last 5 years?"
- "Compare current industrial production to historical data"
- "Show me investment data for the last 10 years"
- "How does consumption compare to pre-pandemic levels?"

## CSV File Format

The application expects CSV files with the following structure:
- A column named `observation_date` with quarterly date values
- A second column with the economic indicator values
- Files should contain approximately 10 years of historical data

## Requirements

- Python 3.8+
- Internet connection (only for first-time model download)
- Microphone for voice input
- Speaker for voice output
