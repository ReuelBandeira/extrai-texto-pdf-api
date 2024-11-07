# Extrai Texto de PDF API

Este projeto utiliza Flask para criar uma API que extrai texto de arquivos PDF. Ele usa várias bibliotecas como `pdf2image`, `pyocr`, `torch`, `transformers`, `pillow`, `opencv-python` e `numpy` para realizar o processamento dos arquivos PDF e extrair o conteúdo.

## Passo 1: Instalar Dependências

Antes de rodar o projeto, instale as dependências necessárias. Se ainda não tiver o Flask e as bibliotecas necessárias, execute o seguinte comando:

```bash
pip install Flask pdf2image pyocr torch transformers pillow opencv-python numpy
```

### Passo 2: Rodar o Servidor Flask

API permite que você envie um arquivo PDF e receba o texto extraído dele. Faça uma requisição POST para o endpoint /extrair_texto com o arquivo PDF anexado.

```bash
 python app.py
```

### Passo 3: Uso da API

A API permite que você envie um arquivo PDF e receba o texto extraído dele. Para isso, faça uma requisição POST para o endpoint /process_pdf, anexando o arquivo PDF no campo file.

```bash
 http://localhost:5000/process_pdf
```
