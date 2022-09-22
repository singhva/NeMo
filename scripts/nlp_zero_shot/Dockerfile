FROM python:3.9

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY ./spacy_app.py .

ENTRYPOINT ["streamlit", "run", "--server.headless", "true"]

CMD ["spacy_app.py"]
