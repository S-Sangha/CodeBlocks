from flask import Flask,render_template, request 
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
    file_image = request.files.get('image')
    print(type(file_image))
    if file_image:
        file_image = file_image.read()
        print("file read")
        #print(file_data)
        # Handle the file data as needed
        return "File uploaded successfully"
    else:
        return "No file provided"

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