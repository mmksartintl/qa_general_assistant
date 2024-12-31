FROM python:3.10

RUN apt-get update && apt-get install -y pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD python3 init_db.py; flask --app main1.py run --host=0.0.0.0


