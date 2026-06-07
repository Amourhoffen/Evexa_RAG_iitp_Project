# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# (Including necessary build tools for FAISS and BM25 if required)
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install google-adk langchain langchain-community faiss-cpu rank_bm25 sentence-transformers

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available for the ADK Server and 8080 for the UI
EXPOSE 8000
EXPOSE 8080

# Define environment variable (Users should pass their own API key at runtime)
ENV GOOGLE_API_KEY=""

# Run the ADK web server when the container launches
CMD ["adk", "web"]
