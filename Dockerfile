FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# torch এবং tensorflow ইনস্টল করা হচ্ছে
RUN pip install torch torchvision torchaudio
RUN pip install tensorflow

COPY . .

ENV FLASK_APP app.py

CMD ["python", "app.py"]
