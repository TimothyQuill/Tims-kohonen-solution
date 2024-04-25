FROM python:3.9-slim
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Make sure a directory for output exists and has the right permissions
RUN mkdir -p /usr/src/app/output
RUN chmod 777 /usr/src/app/output

# Run the program when the container launches
CMD ["python", "main.py"]