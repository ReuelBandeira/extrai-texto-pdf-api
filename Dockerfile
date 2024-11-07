# Use a imagem oficial do Python
FROM python:3.9-slim

# Define o diretório de trabalho no contêiner
WORKDIR /app

# Copie o arquivo de requisitos para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código para o contêiner
COPY . .

# Exponha a porta que o Flask usa
EXPOSE 5000

# Defina o comando de entrada para rodar o aplicativo
CMD ["python", "app.py"]
