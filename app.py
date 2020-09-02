from flask import Flask,request,jsonify,render_template,send_file
import pandas as pd
import os
import numpy as np
import cv2
import pytesseract
from PIL import Image

image_folder = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = image_folder

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict-dogs-cats-api',methods=["POST"])
def predict_api():
    data = request.get_json()
    image = request.files['Image']
    img = Image.open(image)
    img = np.array(img)
    img = cv2.resize(img,dsize=(600,384))

    name_img=img[93:131,6:330]
    name = pytesseract.image_to_string(name_img)

    father_name_img=img[137:167,5:350]
    father_name = pytesseract.image_to_string(father_name_img)

    date_of_birth_img=img[172:222,4:150]
    date_of_birth = pytesseract.image_to_string(date_of_birth_img)

    pan_img=img[243:289,4:233]
    pan = pytesseract.image_to_string(pan_img)

    pan_details = [
        {"Name : ":name},
        {"Father's Name : ":father_name},
        {"Date of Birth : ":date_of_birth},
        {"Pan ID : ":pan}
    ]

    return jsonify(results=pan_details)

@app.route('/get-pan-card-details',methods=["POST"])
def predict_front_end():
    data = request.form.get('filename')
    image = request.files['filename']

    img = Image.open(image)
    img = np.array(img)
    img = cv2.resize(img,dsize=(600,384))

    name_img=img[93:131,6:330]
    name = pytesseract.image_to_string(name_img)

    father_name_img=img[137:167,5:350]
    father_name = pytesseract.image_to_string(father_name_img)

    date_of_birth_img=img[172:222,4:150]
    date_of_birth = pytesseract.image_to_string(date_of_birth_img)

    pan_img=img[243:289,4:233]
    pan = pytesseract.image_to_string(pan_img)

    signature_img=img[299:360,4:260]
    signature_img = cv2.cvtColor(signature_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('static/images' , 'signature.jpg'), signature_img)
    #full_signature = "dynamic\images\signature.jpg"
    full_signature = os.path.join('static/images' , 'signature.jpg')

    photo_img = img[243:371,450:585]
    photo_img = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('static/images' , 'photo.jpg'), photo_img)
    #full_photo = "dynamic\images\photo.jpg"
    full_photo = os.path.join('static/images' , 'photo.jpg')

    pan_details = {
    "Name":name,
    "Father's Name":father_name,
    "Date of Birth":date_of_birth,
    "Pan ID":pan
    }

    return render_template("result.html",prediction_text=pan_details,signature = full_signature, photo = full_photo)



if __name__ == "__main__":
    app.run(debug=True)
