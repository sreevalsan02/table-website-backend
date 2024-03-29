from ultralytics import RTDETR
import cv2
import io
from PIL import Image
from flask import Flask, request, redirect, url_for,send_file
from flask_cors import CORS
app = Flask(__name__)

CORS(app)

@app.route('/upload', methods=['GET', 'POST'])

def table_image():
    if request.method =='POST':
        img_file = request.files['the_file']
        model = RTDETR('best.pt')
        img = Image.open(img_file)
        results = model(img)
        boxes = results[0].boxes
        lis = []
        for i in boxes.xyxy[0]:
            lis.append(i.item())
       

        x = int(lis[0])
        y = int(lis[1])
        x_end = int(lis[2])
        y_end = int(lis[3])

        cropped_img = img.crop((x,y,x_end,y_end))
        
        img_io = io.BytesIO()
        cropped_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
        
    return 'noo'


