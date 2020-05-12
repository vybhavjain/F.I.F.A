from flask import Flask, redirect, render_template, request, session, url_for
import speech_to_text
import os
import image_gen
app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def main_page():
    if request.method == 'GET':
        return render_template("main_page.html")
    else:
        filename = request.form['filename']
        text=speech_to_text.listen2(filename)
        location = os.getcwd()
        image_gen.generate_image(text)
        return render_template("post_recording.html", text=text, location = location, filename=filename)

@app.route('/keyword',methods=["GET","POST"])
def keyword():
    if request.method == 'GET':
        #Do other things to fetch image from model
        location= os.getcwd()
        return render_template("post_recording.html", image_generated=location)
        
    #else:
        #Return a 404 not found page
     #   return render_template("main_page.html")

#banana2
if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)