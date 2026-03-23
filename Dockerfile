FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose both ports
EXPOSE 8050 8000

# Run startup script that starts both services
CMD ["python", "run.py"]