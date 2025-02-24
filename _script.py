# Standard library imports
import os
import platform
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

# Third-party imports
import cv2
import numpy as np
import openpyxl
import pandas as pd
import plotly.express as px
import pyaudio
import pytesseract
import speech_recognition as sr
import streamlit as st
from fpdf import FPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Initialize workbook and audio settings
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
    st.session_state.workbook = openpyxl.Workbook()

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Define LANGUAGES dictionary at the top
LANGUAGES = {
    'English': {
        'title': 'SwapXpert (Data Uploader)',
        'upload_text': 'Upload your files (CSV or Excel):',
        'processing': 'Processing...',
        'success': 'Success!'
    },
    'Urdu': {
        'title': 'سوئپ ایکسپرٹ',
        'upload_text': 'فائلیں اپ لوڈ کریں (CSV یا Excel):',
        'processing': 'پروسیسنگ جاری ہے...',
        'success': 'کامیاب!'
    }
}

# Initialize session state if not exists
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
    st.session_state.workbook = openpyxl.Workbook()

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Move these to the top of your file, after the imports and before the main code
# Language and theme selection in main sidebar
st.sidebar.title("🧠 Laiqely Mind – AI for Smarter Thinking")

# Language selection
language = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()))
texts = LANGUAGES[language]

# Theme selection
# Theme selection
theme = st.sidebar.radio("Theme", ["Dark", "Light"])
if theme == "Light":
    st.markdown("""
        <style>
            /* Main App Background - Dark Gray 600 */
            .stApp, .main, div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {
                background-color: #718096 !important;  /* Dark Gray 600 */
            }
            
            /* Content Container Background */
            .block-container {
                background-color: #7f8ea3 !important;  /* Slightly lighter gray */
                padding: 2rem !important;
                border-radius: 10px !important;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #64748b !important;  /* Darker shade */
                border-right: 2px solid #cc0000;
            }
            
            /* Text colors adjusted for better contrast */
            [data-testid="stSidebar"] .stMarkdown,
            .stMarkdown,
            p,
            label {
                color: white !important;
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: #ff0000 !important;
                font-weight: bold !important;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
            }
            
            .stButton > button {
                background-color: #cc0000 !important;
                color: white !important;
                border: none !important;
                padding: 8px 16px !important;
                border-radius: 5px !important;
                font-weight: bold !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            }
            
            .stButton > button:hover {
                background-color: #aa0000 !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
            }
            
            /* Input Fields with darker background */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stTextArea > div > textarea,
            .stSelectbox > div > div {
                background-color: #4a5568 !important;  /* Even darker gray */
                color: white !important;
                border: 1px solid #cc0000 !important;
                border-radius: 4px !important;
                padding: 8px !important;
            }
            
            /* File Uploader */
            .stUploadedFile {
                background-color: #4a5568 !important;
                border: 1px solid #cc0000 !important;
                border-radius: 4px !important;
                color: white !important;
            }
            
            /* DataFrames with darker background */
            .stDataFrame {
                background-color: #4a5568 !important;
                padding: 10px !important;
                border-radius: 4px !important;
                border: 1px solid #cc0000 !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .stApp { 
                background-color: #121212 !important; 
                color: white !important; 
            }
            
            [data-testid="stSidebar"] {
                background-color: #1a1a1a !important;
                box-shadow: 3px 3px 10px rgba(255, 0, 0, 0.5);
            }
            
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #ff0000 !important;
                text-shadow: 2px 2px 4px rgba(255, 0, 0, 0.7);
            }
            
            .stButton>button {
                background-color: #ff0000 !important;
                color: white !important;
                border-radius: 10px !important;
                box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.7) !important;
            }
            
            .stButton>button:hover {
                background-color: #cc0000 !important;
                box-shadow: 3px 3px 12px rgba(255, 0, 0, 0.9) !important;
            }
            
            .stTextInput>div>div>input, 
            .stTextArea>div>textarea {
                background-color: #222 !important;
                color: white !important;
                border-radius: 8px !important;
                border: 1px solid #ff0000 !important;
                box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.6) !important;
            }
            
            .stFileUploader>div>div>input {
                background-color: #222 !important;
                color: white !important;
                border: 2px solid #ff0000 !important;
                border-radius: 8px !important;
                box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.7) !important;
            }
            
            .stDataFrame {
                background-color: #222 !important;
                color: white !important;
                border: 1px solid #ff0000 !important;
                box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.7) !important;
            }
        </style>
    """, unsafe_allow_html=True)
    # Page selection (move this after theme selection)
page = st.sidebar.radio("Choose an Option:", [
    "SwapXpert (Data Uploader)",
    "AI Assistant",
    "Voice to Text",
    "Handwritten to Text"
])

# 🎨 UI Styling
# 🎨 Custom UI Styling
st.markdown(
    """
    <style>
        .stApp { background-color: #121212; color: white; }
        [data-testid="stSidebar"] {
            background-color: #1a1a1a !important;
            box-shadow: 3px 3px 10px rgba(255, 0, 0, 0.5);
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ff0000 !important;
            text-shadow: 2px 2px 4px rgba(255, 0, 0, 0.7);
        }
        .stButton>button {
            background-color: #ff0000;
            color: white;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.7);
        }
        .stButton>button:hover {
            background-color: #cc0000;
            box-shadow: 3px 3px 12px rgba(255, 0, 0, 0.9);
        }
        .stTextInput>div>div>input, .stTextArea>div>textarea {
            background-color: #222;
            color: white;
            border-radius: 8px;
            border: 1px solid #ff0000;
            box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.6);
        }
        .stFileUploader>div>div>input {
            background-color: #222;
            color: white;
            border: 2px solid #ff0000;
            border-radius: 8px;
            box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.7);
        }
        .stTable { 
            border: 1px solid #ff0000;
            box-shadow: 2px 2px 8px rgba(255, 0, 0, 0.7);
        }
        .block-container {
            padding: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Tesseract configuration
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # For Linux/Cloud deployment
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# 🎤 AI Assistant Function
def ai_response(user_input):
    responses = {
        "hello": "Hello! How can I assist you today? 😊",
        "hi": "Hey there! What can I do for you? 👋",
        "how are you?": "I'm an AI, so I don't have feelings, but I'm here to help you! 🚀",
        "tell me a joke": "Why did the computer catch a cold? Because it left its Windows open! 😂",
        "what is your name": "I am an AI 🤖",
        "assalamoalikum": "Walaikumassalam! How can I assist you?",
        "what is python": "Python is a powerful programming language for web development, AI, and more! 🐍"
    }
    return responses.get(user_input.lower(), "Hmm, I'm not sure about that. But I'd love to help! 😊")

# 🎙 Convert Voice to Text
def convert_voice_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Error connecting to recognition service."

# 🎤 Record Voice Function
def record_voice():
    try:
        # First check if pyaudio is available
        import pyaudio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100
        seconds = 5

        p = pyaudio.PyAudio()
        
        # List all available input devices
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        input_device_index = None

        # Find the first available input device
        for i in range(numdevices):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                input_device_index = i
                break

        if input_device_index is None:
            st.error("No microphone found. Please connect a microphone and try again.")
            p.terminate()
            return None

        try:
            stream = p.open(format=sample_format,
                          channels=channels,
                          rate=fs,
                          input=True,
                          input_device_index=input_device_index,
                          frames_per_buffer=chunk)
            
            # Rest of the recording code remains the same
            with st.spinner("🎤 Recording... Please speak now"):
                frames = []
                for i in range(0, int(fs / chunk * seconds)):
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(temp_audio.name, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))
            wf.close()

            return temp_audio.name

        except Exception as e:
            st.error(f"Error during recording: {str(e)}")
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()
            return None

    except ImportError:
        st.error("PyAudio is not installed. Voice recording is not available in the cloud version.")
        return None
    except Exception as e:
        st.error(f"Error initializing audio: {str(e)}")
        return None

# 📝 Extract Handwritten Text
def extract_text_from_image(image):
    try:
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = image.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)

        open_cv_image = np.array(image)
        if open_cv_image.ndim == 2:
            processed_image = open_cv_image
        else:
            processed_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

        processed_image = cv2.adaptiveThreshold(
            processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        try:
            text = pytesseract.image_to_string(Image.fromarray(processed_image))
            if not text.strip():
                return "No text detected. Please try with a clearer image."
            return text
        except Exception as e:
            st.error(f"Error in text extraction: {str(e)}")
            return "Error in processing the image. Please try again."
            
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return "Error in processing the image. Please try again."

# 📄 Create PDF Function
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes, "F")
    return pdf_bytes.getvalue()

# 🤖 AI Assistant
if page == "AI Assistant":
    st.title("🤖 " + ("AI Assistant" if language == "English" else "مصنوعی ذہانت معاون"))
    user_query = st.text_input("Ask me anything:")
    if st.button("Get Answer"):
        st.write(ai_response(user_query))

# 🎙 Voice to Text
elif page == "Voice to Text":
    st.title("🎙 " + ("Voice to Text Converter" if language == "English" else "آواز سے متن میں تبدیل کریں"))
    
    if st.button("Record Voice"):
        recorded_audio = record_voice()
        if recorded_audio:
            st.success("Recording complete! Now converting to text...")
            text = convert_voice_to_text(recorded_audio)
            st.text_area("Converted Text:", value=text, height=150)

            if st.button("Download as PDF"):
                pdf_bytes = create_pdf(text)
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="voice_text.pdf",
                    mime="application/pdf"
                )

# 📝 Handwritten to Text
elif page == "Handwritten to Text":
    st.title("📝 " + ("Handwritten to Text Converter" if language == "English" else "ہاتھ سے لکھا ہوا متن میں تبدیل کریں"))
    uploaded_image = st.file_uploader("Upload handwritten note (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_image and st.button("Extract Text"):
        image = Image.open(uploaded_image)
        text = extract_text_from_image(image)
        st.text_area("Extracted Text:", value=text, height=150)

        if st.button("Download as PDF"):
            pdf_bytes = create_pdf(text)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="handwritten_text.pdf",
                mime="application/pdf"
            )

# 🔄 SwapXpert (Data Uploader)
# SwapXpert section
if page == "SwapXpert (Data Uploader)":
    st.title(texts['title'])
    
    # File upload
    uploaded_files = st.file_uploader(texts['upload_text'], 
                                    type=["csv", "xlsx"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            try:
                # Load data
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                st.session_state.dataframes[file.name] = df
                st.success(f"File {file.name} loaded successfully!")
                
                # Data Preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head())
                
                # Data Analysis Options
                st.subheader("🔍 Analysis Options")
                analysis_type = st.selectbox(
                    "Choose Analysis Type",
                    ["Basic Analysis", "Advanced Analysis", "Custom Analysis"]
                )
                
                if analysis_type == "Basic Analysis":
                    # Basic Statistics
                    st.write("📈 Statistical Summary:")
                    st.dataframe(df.describe())
                    
                    # Missing Values
                    st.write("❓ Missing Values:")
                    missing = df.isnull().sum()
                    st.dataframe(missing[missing > 0])
                    
                    # Data Types
                    st.write("📋 Data Types:")
                    st.dataframe(df.dtypes)
                
                elif analysis_type == "Advanced Analysis":
                    # Correlation Matrix
                    st.write("🔄 Correlation Matrix:")
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if len(numeric_cols) > 0:
                        corr = df[numeric_cols].corr()
                        fig = px.imshow(corr, color_continuous_scale='RdBu')
                        st.plotly_chart(fig)
                    
                    # Distribution Plots
                    st.write("📊 Distribution Plots:")
                    selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
                    fig = px.histogram(df, x=selected_col)
                    st.plotly_chart(fig)
                
                elif analysis_type == "Custom Analysis":
                    # Column Selection
                    selected_cols = st.multiselect("Select Columns for Analysis", df.columns)
                    if selected_cols:
                        st.write("📊 Selected Columns Preview:")
                        st.dataframe(df[selected_cols])
                        
                        # Custom Statistics
                        st.write("📈 Custom Statistics:")
                        st.dataframe(df[selected_cols].describe())
                        
                        # Custom Plots
                        if len(selected_cols) >= 2:
                            st.write("🔄 Scatter Plot:")
                            x_col = st.selectbox("Select X-axis", selected_cols)
                            y_col = st.selectbox("Select Y-axis", [col for col in selected_cols if col != x_col])
                            fig = px.scatter(df, x=x_col, y=y_col)
                            st.plotly_chart(fig)
                
                # Data Cleaning Options
                st.subheader("🧹 Data Cleaning")
                cleaning_options = st.multiselect(
                    "Select Cleaning Operations",
                    ["Remove Missing Values", "Fill Missing Values", "Remove Duplicates", "Convert Data Types"]
                )
                
                if cleaning_options:
                    if "Remove Missing Values" in cleaning_options:
                        df = df.dropna()
                        st.success("Removed missing values")
                    
                    if "Fill Missing Values" in cleaning_options:
                        fill_method = st.selectbox(
                            "Choose Fill Method",
                            ["Mean", "Median", "Mode", "Custom Value"]
                        )
                        if fill_method == "Custom Value":
                            fill_value = st.text_input("Enter fill value")
                            df = df.fillna(fill_value)
                        else:
                            for col in df.select_dtypes(include=['int64', 'float64']):
                                if fill_method == "Mean":
                                    df[col] = df[col].fillna(df[col].mean())
                                elif fill_method == "Median":
                                    df[col] = df[col].fillna(df[col].median())
                                elif fill_method == "Mode":
                                    df[col] = df[col].fillna(df[col].mode()[0])
                        st.success("Filled missing values")
                    
                    if "Remove Duplicates" in cleaning_options:
                        df = df.drop_duplicates()
                        st.success("Removed duplicates")
                    
                    if "Convert Data Types" in cleaning_options:
                        for col in df.columns:
                            new_type = st.selectbox(
                                f"Convert {col} to:",
                                ["Keep Current", "Integer", "Float", "String", "DateTime"]
                            )
                            if new_type != "Keep Current":
                                try:
                                    if new_type == "Integer":
                                        df[col] = df[col].astype(int)
                                    elif new_type == "Float":
                                        df[col] = df[col].astype(float)
                                    elif new_type == "String":
                                        df[col] = df[col].astype(str)
                                    elif new_type == "DateTime":
                                        df[col] = pd.to_datetime(df[col])
                                except Exception as e:
                                    st.error(f"Error converting {col}: {str(e)}")
                
                # Download Options
                st.subheader("💾 Download Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{file.name.split('.')[0]}_processed.csv",
                        "text/csv"
                    )
                
                with col2:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    st.download_button(
                        "Download Excel",
                        buffer.getvalue(),
                        f"{file.name.split('.')[0]}_processed.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    summary_buffer = BytesIO()
                    with pd.ExcelWriter(summary_buffer, engine='openpyxl') as writer:
                        df.describe().to_excel(writer)
                    st.download_button(
                        "Download Summary",
                        summary_buffer.getvalue(),
                        f"{file.name.split('.')[0]}_summary.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
            except Exception as e:
                st.error(f"Error processing file {file.name}: {str(e)}")
# 🔚 Footer
st.write("---")
st.write("🚀 Built with Streamlit | By Laiqa Eman")

# Add these helper functions if they're not already defined
def suggest_data_cleaning(df):
    suggestions = {
        'recommendations': [],
        'missing_values': {},
        'outliers': {},
        'data_type_suggestions': {}
    }
    
    # Check for missing values
    missing = df.isnull().sum()
    for col in df.columns:
        missing_count = missing[col]
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            suggestions['missing_values'][col] = {
                'count': missing_count,
                'percentage': round(percentage, 2),
                'suggestion': 'Fill with mean/median' if df[col].dtype in ['int64', 'float64'] else 'Fill with mode'
            }
    
    return suggestions

def clean_data(df, options):
    if options.get('drop_null_rows'):
        df = df.dropna(how='all')
    if options.get('drop_null_cols'):
        df = df.dropna(axis=1, how='all')
    if options.get('custom_fill_value'):
        df = df.fillna(options['custom_fill_value'])
    if options.get('columns_to_drop'):
        df = df.drop(columns=options['columns_to_drop'])
    return df

def convert_column_types(df, type_conversions):
    for col, new_type in type_conversions.items():
        try:
            if new_type == 'datetime':
                df[col] = pd.to_datetime(df[col])
            else:
                df[col] = df[col].astype(new_type)
        except Exception as e:
            st.error(f"Error converting {col} to {new_type}: {str(e)}")
    return df

def generate_descriptive_stats(df):
    return df.describe()

def generate_correlation_matrix(df):
    return df.select_dtypes(include=['number']).corr()

def get_value_counts(df, column):
    return df[column].value_counts()

def create_download_link(df, format_type):
    if format_type == 'csv':
        buffer = df.to_csv(index=False).encode()
        mime_type = 'text/csv'
        file_ext = '.csv'
    elif format_type == 'excel':
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_ext = '.xlsx'
    elif format_type == 'json':
        buffer = df.to_json().encode()
        mime_type = 'application/json'
        file_ext = '.json'
    return buffer, mime_type, file_ext
    