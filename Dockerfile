# Base image Python di rakho taaki app.py chal sake
FROM python:3.9-slim

# Working directory set karo
WORKDIR /app

# Saari files (app.py samet) container vich copy karo
COPY . .

# Je koi library (openai etc.) install karni hai taan:
# RUN pip install openai

# Container start hunde hi app.py chalao
CMD ["python", "app.py"]# Base image Python di rakho taaki app.py chal sake
FROM python:3.9-slim

# Working directory set karo
WORKDIR /app

# Saari files (app.py samet) container vich copy karo
COPY . .

# Je koi library (openai etc.) install karni hai taan:
# RUN pip install openai

# Container start hunde hi app.py chalao
CMD ["python", "app.py"]