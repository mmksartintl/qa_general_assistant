FROM python:3.10

RUN apt-get update && apt-get install -y pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "main.py"]


