import os
import sys
from typing import List

sys.path.append(os.getcwd())

import numpy
import torch
import PIL.Image

from tha2.poser.poser import Poser, PoseParameterCategory, PoseParameterGroup
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, \
    convert_output_image_from_torch_to_numpy
from flask import Flask, request, Response, jsonify, render_template
from flask_restx import Api, Resource, reqparse, fields
from flask_cors import CORS
from flask import send_file
from PIL import Image

import zipfile
import io
import pathlib

import cv2

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = 'D:/ouput/'
CORS(app, supports_credentials=True)  # 다른 포트번호에 대한 보안 제거
api = Api(app)


class ControlPose:
    def __init__(self,  poser: Poser, device: torch.device, image):

        self.alpha = 1
        self.device = device
        self.poser = poser
        self.pil_image = image

    # sad
    def make_sad(self):
        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]
        current_pose[0] = self.alpha
        current_pose[1] = self.alpha
        print("sad")
        self.save_img('sad', current_pose)  # 여기서부터 막힘 save image
        print("dddd")

    # happy
    def make_happy(self):

        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]
        current_pose[14] = self.alpha
        current_pose[15] = self.alpha
        current_pose[29] = self.alpha

        self.save_img('happy', current_pose)

    # wink
    def make_wink(self):

        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]

        current_pose[14] = self.alpha
        self.save_img('wink', current_pose)

    # angry
    def make_angry(self):
        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]

        current_pose[2] = self.alpha
        current_pose[3] = self.alpha

        self.save_img('angry', current_pose)

    # tired
    def make_tired(self):
        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]

        current_pose[20] = self.alpha
        current_pose[21] = self.alpha

        self.save_img('tired', current_pose)

    def save_img(self, save_name, current_pose):
        pose = torch.tensor(current_pose, device=self.device)
        print("11")
        output_index = 0
        print("11")

        ########
        torch_source_image = extract_pytorch_image_from_PIL_image(self.pil_image).to(self.device)

        print("kk")
        output_image = self.poser.pose(torch_source_image, pose, output_index)[0].detach().cpu()
        print("11")
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))

        last_output_numpy_image = numpy_image

        print("save_name")
        numpy_image = last_output_numpy_image
        print("save_name")
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        # print("save_name")
        # os.makedirs(os.path.dirname(save_name), exist_ok=True)      # 디렉토리 생성
        print(save_name)
        # pil_image.open(save_name)
        pil_image.save('D:/ouput/' + save_name + '.png')  # 여기서 에러


@app.route('/')
def uploader():
    return "ok"


@app.route('/anime', methods=['POST'])
def Test():
    cuda = torch.device('cuda')
    import tha2.poser.modes.mode_20

    poser = tha2.poser.modes.mode_20.create_poser(cuda)

    print("kkkkk")

    file = request.files['file']
    print(file.filename)
    print(os.path.join(app.config['UPLOAD_FOLDER']))
    print(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    image = Image.open('D:/ouput/' + file.filename)       #  'C:/Users/DS/DualStyleGAN/output/' + save_name + '.png'
    print(image.info)

    ## 여기서 사진 작업 해주기

    print('Load models successfully!')

    print("이미지명: ",image)

    import cv2

    ## emoticon에 맞게 resize하는 코드
    img = cv2.imread('D:/ouput/' + file.filename)  # 삽입될 이미지
    img = cv2.resize(img, (128, 128))  # 삽입될 이미지 리사이즈


    back_img = cv2.imread('D:/ouput/square.png')  # 배경 이미지(256x256 흰화면)
    x_offset = y_offset = 50
    back_img[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
    cv2.imwrite('D:/ouput/img.png', back_img)  # 저장


    import PIL.Image
    rgb_image = PIL.Image.open('D:/ouput/img.png')
    rgba_image = rgb_image.convert('RGBA')  # 알파 채널 추가해주기

    rgba_image.save('D:/ouput/rgba_image.png')


    ## square 배경 제거
    import numpy as np
    import cv2

    imageUrl = 'D:/ouput/rgba_image.png'
    img = cv2.imread(imageUrl)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, 256, 256)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    cv2.imwrite("D:/ouput/result-remove.png", dst)
    cv2.waitKey(0)

    print("error1")


    image = Image.open('D:/ouput/result-remove.png')

    print("error2")
    ## 여기서 이모티콘 생성 + 이모티콘 저장
    ## 수치 조절
    ctest = ControlPose( poser, cuda, image)
    print("saddddd")
    ctest.make_sad()
    ctest.make_angry()
    ctest.make_happy()
    ctest.make_wink()
    ctest.make_tired()

    print('Generate images successfully!')

    ## 이모티콘 response

    base_path = pathlib.Path('D:/ouput/')
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode='w') as z:
        for f_name in base_path.iterdir():
            z.write(f_name)
    data.seek(0)
    return send_file(
        data,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='data.zip'
    )

    # return send_file('D:/ouput/' + 'happy.png')
    # return "ok"


if __name__ == "__main__":
    app.run()
