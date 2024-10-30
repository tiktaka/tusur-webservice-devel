FROM python:3.12-slim

WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY flaskapp /flaskapp
WORKDIR /flaskapp

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:5000", "wsgi:app"]