from flask import Flask,request,jsonify,render_template,send_file
#from flask_dropzone import Dropzone
#from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import pandas as pd
import os
import numpy as np
import cv2
import pytesseract
from PIL import Image
from random import randint
import datetime
import time

#image_folder = os.path.join('static', 'images')
image_folder = "/static"

app = Flask(__name__)
# dropzone = Dropzone(app)
app.config['UPLOAD_FOLDER'] = image_folder

# Dropzone settings
# app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
# app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
# app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
# app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
#app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
# app.config['UPLOADED_PHOTOS_DEST'] = os.path.join('public', 'images')
# photos = UploadSet('photos', IMAGES)
# configure_uploads(app, photos)
# patch_request_class(app)  # set maximum file size, default is 16MB

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
    #ran = randint(0,10e6)
    #signature_path = 'signature.jpg?dummy='+str(ran)
    #full_signature = os.path.join(app.config['UPLOAD_FOLDER'],signature_path)
    signature_path = "signature"+str(time.time())+".jpg"
    full_signature = os.path.join('static',signature_path)
    #full_signature = os.path.join(app.config['UPLOAD_FOLDER'], 'signature.jpg')
    #os.remove(full_signature)
    cv2.imwrite(full_signature,signature_img)
    #cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'signature.jpg'),signature_img)
    #cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'signature.jpg'), signature_img)
    #cv2.imwrite(os.path.join('static/images' , 'signature.jpg'), signature_img)
    #full_signature = os.path.join('static/images' , 'signature.jpg')

    photo_img = img[243:371,450:585]
    photo_img = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)
    #photo_path = 'photo.jpg?dummy='+str(ran)
    #full_photo = os.path.join(app.config['UPLOAD_FOLDER'],photo_path)
    photo_path = "photo"+str(time.time())+".jpg"
    full_photo = os.path.join('static',photo_path)
    #full_photo = os.path.join(app.config['UPLOAD_FOLDER'], 'photo.jpg')
    #os.remove(full_photo)
    cv2.imwrite(full_photo,photo_img)
    #cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'photo.jpg'),photo_img)
    #cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'photo.jpg'), photo_img)
    #cv2.imwrite(os.path.join('static/images' , 'photo.jpg'), photo_img)
    #full_photo = os.path.join(app.config['UPLOAD_FOLDER'],'photo.jpg')
    #full_photo = os.path.join('static/images' , 'photo.jpg')

    pan_details = {
    "Name":name,
    "Father's Name":father_name,
    "Date of Birth":date_of_birth,
    "Pan ID":pan
    }

    return render_template("result.html",prediction_text=pan_details,signature = signature_path, photo = photo_path)

# def delete_files():
#     os.remove(os.path.join(image_folder,"photo.jpg"))
#     os.remove(os.path.join(image_folder,"signature.jpg"))

if __name__ == "__main__":
    app.run(debug=True)
    #delete_files()
