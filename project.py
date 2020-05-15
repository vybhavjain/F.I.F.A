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
        print(filename)
        text=speech_to_text.listen2(filename)
        location = os.getcwd()
        text_lower = text.lower()
        image_gen.generate_image(text_lower)
        return render_template("post_recording.html", text=text_lower, location = location, filename=filename)

@app.route('/keyword',methods=["GET","POST"])
def keyword():
    if request.method == 'GET':
        #Fetch image from directory
        filename=request.args.get("text")
        print(filename)
        return render_template("post_recording.html")
        
    #else:
        #Return a 404 not found page
     #   return render_template("main_page.html")

#banana2
if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)