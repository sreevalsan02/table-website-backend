from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pdf2image import convert_from_path


app = Flask(__name__)
# CORS(app)

@app.route('/',methods = ['GET'])
def hello():
    return "hello"

@app.route('/trial',methods = ['GET','POST'])

def fun_trial():
    if request.method == 'POST':
        img_file = request.files['the_file']
        
        if img_file.filename.endswith('.pdf'):
            pdf_file_name = secure_filename(img_file.filename)
            img_file.save(pdf_file_name)
            img_file = convert_from_path(pdf_file_name)
            img_file[0].show()
        return "ok"
    return "noo"

if __name__ =="__main__":
    app.run()
