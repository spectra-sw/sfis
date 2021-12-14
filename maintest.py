from flask import Flask, render_template, Response
import urllib.request
 
def get_frame():
    url = 'http://localhost:8000/WEB_77TX8ds2du4GFiL/video_feed'
    with urllib.request.urlopen(url) as url:
        img = url.read()
        return img 
     
app = Flask(__name__)   
 
@app.route('/')
def index():
    return render_template('test.html')

def gen():
    while True:
        frame = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
@app.route('/video_streaming')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)


