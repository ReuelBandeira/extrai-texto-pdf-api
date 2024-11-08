from flask import Flask, request, jsonify
from take_pages import PdfIT
import os

app = Flask(__name__)

# Configurações de upload de arquivo
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Função para verificar a extensão permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Processar o PDF com a classe PdfIT
        pdf_it = PdfIT()
        list_it, list_posto, list_titulo, list_pagina = pdf_it.get(filename)

        # Combinar list_posto, list_titulo e list_pagina em uma lista de dicionários
        resultado = [
            {"pagina": pagina, "posto": posto, "titulo_posto": titulo}
            for pagina, posto, titulo in zip(list_pagina, list_posto, list_titulo)
        ]

        # Retornar o resultado em formato JSON
        return jsonify(resultado)
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)


