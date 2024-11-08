from pdf2image import convert_from_path
import numpy as np
import cv2
from PIL import Image
import torch
import pyocr.builders
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings
warnings.simplefilter('ignore')

class PdfIT:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device)
        tools = pyocr.get_available_tools()
        self.tool = tools[0]
        self.image_it = cv2.imread('templates/IT.png')
        self.image_posto = cv2.imread('templates/posto.png')
        self.image_title_posto = cv2.imread('templates/titulo_posto.png')

    @staticmethod
    def convert_pdf(filename):
        paginas_pil = convert_from_path(filename, dpi=300)
        images_paginas = [np.array(pagina) for pagina in paginas_pil]
        return images_paginas

    @staticmethod
    def nms_adaptively(boxes):
        result = []
        if not boxes:
            return result
        for i, box in enumerate(boxes):
            if i == 0 or all(abs(box[0][0] - res[0][0]) >= 20 or abs(box[0][1] - res[0][1]) >= 20 for res in result):
                result.append(box)
        return result

    @staticmethod
    def get_area_util(image_input):
        image_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) if len(image_input.shape) == 3 else image_input
        _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        if np.all(image_binary == 255):
            roi_pads_gerber = (0, 0, image_binary.shape[1], image_binary.shape[0])
        else:
            coord_black = np.where(image_binary == 0)
            roi_left, roi_right = np.min(coord_black[1]), np.max(coord_black[1])
            roi_up, roi_down = np.min(coord_black[0]), np.max(coord_black[0])
            roi_pads_gerber = (roi_left, roi_up, roi_right, roi_down)
        return roi_pads_gerber

    def template_matching(self, image_input, image_template_input):
        image_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        image_template = cv2.cvtColor(image_template_input, cv2.COLOR_BGR2GRAY)
        y, x = image_template.shape[0], image_template.shape[1]
        img_matching = cv2.matchTemplate(image_gray, image_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(img_matching >= 0.8)
        result = [[pt, img_matching[pt[1], pt[0]], [x, y]] for pt in zip(*loc[::-1])]
        return self.nms_adaptively(result)

    def ocr_trocr(self, image_input):
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def ocr_pyocr(self, image_input):
        text = self.tool.image_to_string(Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)),
                                         builder=pyocr.builders.TextBuilder())
        return [x for x in text.splitlines() if x.strip()]

    def find_template(self, image_input):
        roi_info = self.get_area_util(image_input)
        x_length, y_length = roi_info[2] - roi_info[0], roi_info[3] - roi_info[1]
        ratio_x, ratio_y = x_length / 3381, y_length / 2276

        template1 = cv2.resize(self.image_it, (0, 0), fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)
        template2 = cv2.resize(self.image_posto, (0, 0), fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)
        template3 = cv2.resize(self.image_title_posto, (0, 0), fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)

        boxe1 = self.template_matching(image_input, template1)
        boxe2 = self.template_matching(image_input, template2)
        boxe3 = self.template_matching(image_input, template3)

        result_it = bool(boxe1)
        result_posto = result_titulo_posto = False
        posto = titulo_posto = None

        if result_it:
            if boxe2:
                x1_posto, y1_posto = boxe2[0][0][0], boxe2[0][0][1] + boxe2[0][2][1]
                x2_posto, y2_posto = x1_posto + boxe2[0][2][0], y1_posto + int(boxe2[0][2][1] * 2.5)
                image_result_posto = image_input[y1_posto:y2_posto, x1_posto:x2_posto]
                result_ocr_posto = self.ocr_trocr(image_result_posto)
                if result_ocr_posto:
                    result_posto = True
                    posto = result_ocr_posto[0]

            if boxe3:
                x1_titulo = boxe3[0][0][0] - int(boxe3[0][2][0] * 1.5)
                y1_titulo = boxe3[0][0][1] + boxe3[0][2][1]
                x2_titulo = boxe3[0][0][0] + int(boxe3[0][2][0] * 2.5)
                y2_titulo = y1_titulo + int(boxe3[0][2][1] * 2.5)
                image_result_titulo_posto = image_input[y1_titulo:y2_titulo, x1_titulo:x2_titulo]
                result_ocr_titulo_posto = self.ocr_pyocr(image_result_titulo_posto)
                if result_ocr_titulo_posto:
                    result_titulo_posto = True
                    titulo_posto = " ".join(result_ocr_titulo_posto)

        return {
            "status_it": result_it,
            "status_posto": result_posto,
            "status_titulo": result_titulo_posto,
            "image": image_input if result_it else None,
            "posto": posto,
            "titulo_posto": titulo_posto
        }

    def get(self, file_pdf):
        paginas = self.convert_pdf(file_pdf)
        list_it, list_posto, list_titulo, list_pagina = [], [], [], []

        for index, pagina in enumerate(paginas):
            print(f'\nPagina: {index + 1}')
            results = self.find_template(pagina)
            print(f'Instrução de trabalho: {results["status_it"]}')
            print(f'Posto: {results["posto"]}')
            print(f'Titulo do posto: {results["titulo_posto"]}')

            if results["status_it"]:
                list_it.append(results["image"])
                list_posto.append(results["posto"])
                list_titulo.append(results["titulo_posto"])
                list_pagina.append(index + 1)  # Guarda o número da página

        return list_it, list_posto, list_titulo, list_pagina
