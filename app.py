from flask import Flask,render_template, request,jsonify
import cv2
import numpy as np


file_image = None
app = Flask(__name__,template_folder="templates") 
  
@app.route("/") 
def hello(): 
    print("server is starting ok")
    return render_template('index.html') 

@app.route("/q1")
def q1():
    print("q1 loading")
    return render_template('solution_upload.html')

@app.route("/tutorial")
def tutorial():
    print("tutorial loading")
    return render_template('tutorial.html')

@app.route('/uploadimage', methods=['POST'])
def upload_image():
    print("uploading image")
    global file_image
    #print(request.files.get('image'))
    fileraw = request.files.get('image')
    file_image = cv2.imdecode(np.frombuffer(fileraw.read(), np.uint8), cv2.IMREAD_COLOR)
    print(type(file_image))

    return "1"
    

@app.route('/q1answer')
def q1a():
    datashow = {
        "input":1,
        "code":"for i in range",
        "output":2
    }
    return render_template('answer.html',data = datashow)

def compute():
    print("compute")
    
compute()
#print(file_image)

if __name__ == '__main__': 
    app.run(debug=True) 