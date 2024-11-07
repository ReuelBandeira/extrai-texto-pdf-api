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
        list_it, list_posto, list_titulo = pdf_it.get(filename)

        # Retornar o resultado em formato JSON
        return jsonify({
            'postos': list_posto,
            'titulo_postos': list_titulo
        })
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
