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
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed', )
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device)
        tools = pyocr.get_available_tools()
        self.tool = tools[0]
        self.image_it = cv2.imread('templates/IT.png')
        self.image_posto = cv2.imread('templates/posto.png')
        self.image_title_posto = cv2.imread('templates/titulo_posto.png')


    @staticmethod
    def convert_pdf(filename):

        paginas_pil = convert_from_path(filename, dpi=300)
        images_paginas = list()

        for i, pagina in enumerate(paginas_pil):
            images_paginas.append(np.array(pagina))

        return images_paginas

    @staticmethod
    def nms_adaptively(boxes):
        result = []
        if len(boxes) == 0:
            return result
        else:
            for i in range(len(boxes)):
                if i == 0:
                    result.append(boxes[i])
                else:
                    aux = 0
                    for j in range(len(result)):
                        if ((abs(boxes[i][0][0] - result[j][0][0]) < 20)
                                and (abs(boxes[i][0][1] -
                                         result[j][0][1]) < 20)):
                            aux += 1
                    if aux == 0:
                        result.append(boxes[i])
            for i in range(len(boxes)):
                for j in range(len(result)):
                    if ((abs(boxes[i][0][0] - result[j][0][0]) < 20) and
                            (abs(boxes[i][0][1] - result[j][0][1]) < 20)):
                        if boxes[i][1] > result[j][1]:
                            result[j] = boxes[i]
            return result

    @staticmethod
    def get_area_util(image_input):

        if len(image_input.shape) == 3:
            image_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_input.copy()

        _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

        if np.all(image_binary == 255):

            roi_pads_gerber = (0, 0, image_binary.shape[1], image_binary.shape[0])

        else:

            coord_black = np.where(image_binary == 0)

            roi_left = coord_black[1][np.argmin(coord_black[1])]
            roi_right = coord_black[1][np.argmax(coord_black[1])]
            roi_up = coord_black[0][np.argmin(coord_black[0])]
            roi_down = coord_black[0][np.argmax(coord_black[0])]

            roi_pads_gerber = (roi_left, roi_up, roi_right, roi_down)

        return roi_pads_gerber

    def template_matching(self, image_input, image_template_input):

        image_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        image_template = cv2.cvtColor(image_template_input, cv2.COLOR_BGR2GRAY)

        y, x = image_template.shape[0], image_template.shape[1]

        img_matching = cv2.matchTemplate(image_gray, image_template,
                                         cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(img_matching >= threshold)

        result = []
        for pt in zip(*loc[::-1]):
            result.append([pt, img_matching[pt[1], pt[0]], [x, y]])

        boxes = self.nms_adaptively(result)

        return boxes

    def ocr_trocr(self, image_input):

        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        pixel_values = self.processor(image, return_tensors='pt').pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def ocr_pyocr(self, image_input):

        text = self.tool.image_to_string(Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)),
                                    builder=pyocr.builders.TextBuilder())
        result_text = [x for x in text.splitlines() if x.strip() != '']

        return result_text

    def find_template(self, image_input):

        roi_info = self.get_area_util(image_input)

        x_length = roi_info[2] - roi_info[0]
        y_length = roi_info[3] - roi_info[1]

        x_base, y_base = 3381, 2276

        ratio_x = x_length / x_base
        ratio_y = y_length / y_base

        template1 = cv2.resize(self.image_it, (0, 0),
                              fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)

        template2 = cv2.resize(self.image_posto, (0, 0),
                              fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)

        template3 = cv2.resize(self.image_title_posto, (0, 0),
                              fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_NEAREST)

        boxe1 = self.template_matching(image_input, template1)
        boxe2 = self.template_matching(image_input, template2)
        boxe3 = self.template_matching(image_input, template3)

        if len(boxe1) != 0:

            image_pagina = image_input
            result_it = True

            if len(boxe2) != 0:

                x1_posto = boxe2[0][0][0]
                y1_posto = boxe2[0][0][1] + boxe2[0][2][1]
                x2_posto = x1_posto + boxe2[0][2][0]
                y2_posto = y1_posto + int(boxe2[0][2][1]*2.5)

                image_result_posto = image_input[y1_posto:y2_posto, x1_posto:x2_posto]
                result_ocr_posto = self.ocr_trocr(image_result_posto)

                if len(result_ocr_posto) != 0:
                    result_posto = True
                    posto = result_ocr_posto[0]

                else:
                    result_posto = False
                    posto = None

            else:
                result_posto = False
                posto = None

            if len(boxe3) != 0:

                x1_titulo = boxe3[0][0][0] - int(boxe3[0][2][0]*1.5)
                y1_titulo = boxe3[0][0][1] + boxe3[0][2][1]
                x2_titulo = boxe3[0][0][0] + int(boxe3[0][2][0]*2.5)
                y2_titulo = y1_titulo + int(boxe3[0][2][1]*2.5)

                image_result_titulo_posto = image_input[y1_titulo:y2_titulo, x1_titulo:x2_titulo]
                result_ocr_titulo_posto = self.ocr_pyocr(image_result_titulo_posto)

                if len(result_ocr_titulo_posto) != 0:
                    result_titulo_posto = True
                    titulo_posto = " ".join(result_ocr_titulo_posto)

                else:
                    result_titulo_posto = False
                    titulo_posto = None

            else:
                result_titulo_posto = False
                titulo_posto = None

        else:
            image_pagina = []
            result_it = False
            result_posto = False
            result_titulo_posto = False
            posto = None
            titulo_posto = None

        dict_result= {
            "status_it": result_it,
            "status_posto": result_posto,
            "status_titulo": result_titulo_posto,
            "image": image_pagina,
            "posto": posto,
            "titulo_posto": titulo_posto
        }

        return dict_result

    def get(self, file_pdf):

        paginas = self.convert_pdf(file_pdf)

        list_it = list()
        list_posto = list()
        list_titulo = list()

        for index, pagina in enumerate(paginas):

            print(f'\nPagina: {index + 1}')
            results = self.find_template(pagina)
            print(f'Instrução de trabalho: {results["status_it"]}')
            print(f'Posto: {results["posto"]}')
            print(f'Titulo do posto: {results["titulo_posto"]}')

            # vi.show_image([pagina])

            if results["status_it"]:
                list_it.append(results["image"])
                list_posto.append(results["posto"])
                list_titulo.append(results["titulo_posto"])


        return list_it, list_posto, list_titulo