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
    print("q3 loading")
    question = {
        "q":"Write a program to calculate how many minutes are in a given number of hours",
        "number":"Hours to Minutes",
        "void":"void"
        }
    return render_template('solution_upload.html',question = question)


@app.route("/q2")
def q2():
    print("q loading")
    question = {
        "q":"Write a program which increments (adds 1 to) your favourite number",
        "number":"Increment favourite number",
        "void":"void"
        }
    return render_template('solution_upload.html',question = question)
    
@app.route("/q3")
def q3():
    print("q1 loading")
    question = {
        "q":"Write a program to check if a number is divisible by 5. If so, divide the input by 5. Otherwise, return the input without changing it.",
        "number":"If statements",
        "void":"void"
        }
    return render_template('solution_upload.html',question = question)
    
@app.route("/q4")
def q4():
    print("q1 loading")
    question = {
        "q":"Write a program which checks if the input is larger than 2. If so, take 1 away and print the result. Otherwise, add 1 and print the result",
        "number":"Adjust and Print",
        "void":"void"
        }
    return render_template('solution_upload.html',question = question)

@app.route("/q5")
def q5():
    print("q1 loading")
    question = {
        "q":"Write a program which prints the numbers from 1-20 divisible by 3",
        "number":"Multiples of 3",
        "void":"void"
        }
    return render_template('solution_upload.html',question = question)
@app.route("/q6")
def q6():
    print("q1 loading")
    question = {
        "q":"Use a while loop to print all numbers less than 30 that aren't divisible by 5",
        "number":"Final challenge: while loops!",
        "void":"void"
        }
    return render_template('solution_upload.html',question = question)
@app.route("/playground")
def sandbox():
    print("q1 loading")
    question = {
        "q":"Experiment with test programs! Enter test inputs too to see how your program behaves",
        "number":"PlayGround",
        "void":"void"
        }
    return render_template('solution_upload.html',question = question)
@app.route("/tutorial")
def tutorial():
    print("tutorial loading")
    return render_template('tutorial.html')

@app.route('/uploadimage', methods=['POST'])
def upload_image():
    print("uploading image")
    global file_image1
    #print(request.files.get('image'))
    fileraw = request.files.get('image')
    file_image1 = cv2.imdecode(np.frombuffer(fileraw.read(), np.uint8), cv2.IMREAD_COLOR)
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
    

if __name__ == '__main__': 
    app.run(debug=True) 