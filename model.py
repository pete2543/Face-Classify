# -*- coding: utf-8 -*-
#from crypt import methods
#from pyexpat import model
#from re import A
from flask import Flask ,render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
 
dic = {0:'ยุโรป',1:'M2',2:'M3',3:'M4',4:'M5'}

model = load_model('project.h5')


model.make_predict_function();

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i)
    i = i.reshape(1, 224, 224, 3)
    predict_x = model.predict(i)
    classes_x = np.argmax(predict_x, axis=1)
    return dic[classes_x[0]]

#routes
@app.route("/",methods=['GET','POST'])
def  main():
   return render_template("index.html",myname="HELLO")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path, prediction2=p.split(" "))

if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
