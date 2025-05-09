FROM python:3.11-slim

WORKDIR /app  # Set working directory inside the container

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory
COPY Chatbot.py .
COPY doc_links.yaml .
COPY pages/ ./pages/

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app using Chatbot.py
CMD ["streamlit", "run", "Chatbot.py"]
