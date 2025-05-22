# Whisper-BERT-Fraud-Detection

## Project Description

This project implements a real-time audio fraud detection system using advanced speech recognition and natural language processing techniques. It captures live audio from the microphone, transcribes it to text using OpenAI's Whisper model, and then analyzes the transcribed text with a pre-trained Arabic BERT model to detect signs of potential financial fraud.

If the text sentiment is detected as negative or suspicious, the system flags the call as a possible fraud attempt; otherwise, it marks it as safe.

## Features

- Real-time audio capture using PyAudio  
- High-accuracy speech-to-text transcription with Whisper  
- Sentiment and fraud keyword analysis using an Arabic BERT model (CAMeL-Lab)  
- Offline processing without reliance on external APIs  
- Console output indicating fraud detection results

## Requirements

- Python 3.7 or higher  
- PyTorch  
- OpenAI Whisper  
- PyAudio  
- Transformers library (Hugging Face)  
- numpy

## Usage

1. Install required packages (preferably via `requirements.txt`):  
   ```bash
   pip install -r requirements.txt
