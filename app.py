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

    name_img=img[93:135,3:341]
    #name_img=img[93:131,6:330]
    name = pytesseract.image_to_string(name_img)
    
    father_name_img=img[140:179,3:341]
    #father_name_img=img[137:167,5:350]
    father_name = pytesseract.image_to_string(father_name_img)
    
    date_of_birth_img=img[181:221,3:341]
    #date_of_birth_img=img[172:222,4:150]
    date_of_birth = pytesseract.image_to_string(date_of_birth_img)
    
    pan_img=img[248:297,1:282]
    #pan_img=img[243:289,4:233]
    pan = pytesseract.image_to_string(pan_img)

    pan_details = [
        {"Name : ":name},
        {"Father's Name : ":father_name},
        {"Date of Birth : ":date_of_birth},
        {"Pan ID : ":pan}
    ]

    return jsonify(results=pan_details)

@app.route('/get-document-details',methods=["POST"])
def predict_front_end():
    data = request.form.get('filename')
    image = request.files['filename']

    selection = request.form.get('document')

    img = Image.open(image)
    img = np.array(img)

    # For PAN Card
    if(selection=="pan"):
        img = cv2.resize(img,dsize=(600,384))

        name_img=img[93:135,3:341]
        name = pytesseract.image_to_string(name_img)

        father_name_img=img[140:179,3:341]
        father_name = pytesseract.image_to_string(father_name_img)

        date_of_birth_img=img[181:221,3:341]
        date_of_birth = pytesseract.image_to_string(date_of_birth_img)

        pan_img=img[248:297,1:282]
        pan = pytesseract.image_to_string(pan_img)

        photo_img = img[239:380,452:591]
        photo_img = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)
        photo_path = "photo"+str(time.time())+".jpg"
        full_photo = os.path.join('static',photo_path)
        cv2.imwrite(full_photo,photo_img)

        signature_img = img[302:351,1:312]
        signature_img = cv2.cvtColor(signature_img, cv2.COLOR_BGR2RGB)
        signature_path = "signature"+str(time.time())+".jpg"
        full_signature = os.path.join('static',signature_path)
        cv2.imwrite(full_signature,signature_img)

        pan_details = {
        "Name":name,
        "Father's Name":father_name,
        "Date of Birth":date_of_birth,
        "Pan ID":pan
        }

        r = render_template("pan.html",prediction_text=pan_details,signature = signature_path,photo = photo_path)

    elif(selection=="voter"):
        img = cv2.resize(img,dsize=(2004,3368))

        name_img = img[2122:2422,895:1943]
        name = pytesseract.image_to_string(name_img)

        father_name_img=img[2706:2945,895:1949]
        father_name = pytesseract.image_to_string(father_name_img)

        date_of_birth_img=img[3172:3334,910:1943]
        date_of_birth = pytesseract.image_to_string(date_of_birth_img)

        voter_id_img=img[906:1106,66:1028]
        voter_id = pytesseract.image_to_string(voter_id_img)

        sex_img=img[2956:3145,1182:1400]
        sex = pytesseract.image_to_string(sex_img,lang='eng',config='--psm 6')

        photo_img=img[922:1878,1177:1927]
        photo_img = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)
        photo_path = "photo"+str(time.time())+".jpg"
        full_photo = os.path.join('static',photo_path)
        cv2.imwrite(full_photo,photo_img)

        voter_details = {
        "Voter ID":voter_id,
        "Name":name,
        "Father's Name":father_name,
        "Sex":sex,
        "Date of Birth":date_of_birth
        }

        r = render_template("voter.html",prediction_text=voter_details,photo = photo_path)

    #r = add_header(r)
    return r

# def delete_files():
#     os.remove(os.path.join(image_folder,"photo.jpg"))
#     os.remove(os.path.join(image_folder,"signature.jpg"))

if __name__ == "__main__":
    app.run(debug=True)
    #delete_files()
