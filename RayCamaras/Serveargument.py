from ray.util.queue import Queue
from fastapi import FastAPI
from datetime import datetime as dt
from scipy.spatial import Delaunay, Voronoi
from fastapi.responses import StreamingResponse, Response, FileResponse, HTMLResponse
#from starlette.responses import StreamingResponse, FileResponse, Response, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
#from fastapi.responses import HTMLResponse, StreamingResponse, Response
from PIL import Image
from ray import serve
import io
import threading as thr
import elasticsearch as ES
import numpy as np
import requests
import imutils
import base64
import aiohttp
import asyncio
import argparse
import time
import json
import cv2
import ray
#----------------------------------------------------------------------------------------
import nest_asyncio
nest_asyncio.apply()
# requirements.txt-----------------------------------------------------------------------
#pip3 freeze > requirements.txt
# Argumento de datos---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Setting")
parser.add_argument('--id', type=str, default='')
parser.add_argument('--source', type=ascii, default='')
parser.add_argument('--frame', type=int, default=10)
parser.add_argument('--urlservices', type=str, default='')
parser.add_argument('--resolution', type=int, default=720)
parser.add_argument('--timeml', type=float, default=0.5)
parser.add_argument('--indexread', type=str, default='')
parser.add_argument('--indexwrite', type=str, default='')
parser.add_argument('--namespace', type=str, default='serve')
parser.add_argument('--nreplica', type=int, default=1)
parser.add_argument('--address', type=str, default='auto')
parser.add_argument('--hostname',type=str, default='localhost')
parser.add_argument('--port', type=int, default=9200)
parser.add_argument('--sizeread', type=int, default=1)
parser.add_argument('--sizeface', type=int, default=50)
parser.add_argument('--desenfoque', type=float, default=0.0)
parser.add_argument('--thr', type=float, default=0.905)
parser.add_argument('--thrcw', type=float, default=35.0)
parser.add_argument('--thrch', type=float, default=35.5)
parser.add_argument('--thrperson', type=float, default=70.0)
parser.add_argument('--thrminperson', type=float, default=40.0)
parser.add_argument('--remove', type=bool, default=False)
parser.add_argument('--add', type=bool, default=False)
args = parser.parse_args()
# json -----------------------------------------------------------------------------------
params  = { 
        "idc":args.id, "thr":args.thr, "thrcw":args.thrcw, "thrch":args.thrch, "size":args.sizeface, "desf":args.desenfoque, 
        "indexread":args.indexread, "indexwrite":args.indexwrite,"host":args.hostname, "port": args.port,
        "sizeread": args.sizeread, "thrperson": args.thrperson, "thrminperson": args.thrminperson
        } 
params  = json.dumps(params).encode('utf-8')
# json view ------------------------------------------------------------------------------
vparams = { 
        "idc":args.id, "thr":0.95, "thrcw":180.0, "thrch":180.0, "size":25, "desf":0.0, 
        "indexread":args.indexread, "indexwrite":args.indexwrite,"host":args.hostname, "port": args.port,
        "sizeread": args.sizeread, "thrperson": args.thrperson, "thrminperson": args.thrminperson
         }
vparams = json.dumps(vparams).encode('utf-8')
# ----------------------------------------------------------------------------------------
inf = {
       "id":args.id, "frame":args.frame, "resolution":args.resolution,"thr":args.thr,
       "size":args.sizeface, "thr_max_people":args.thrperson, "thr_min_people": args.thrminperson
      }
# ----------------------------------------------------------------------------------------
NCAMRA = 'CAM'
NQUERY = 'QRY'
QFRAME = 'QFR'
QQUERY = 'QQR'
TTESTY = 'TTS'
# ----------------------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
# ----------------------------------------------------------------------------------------
ray.init(address=args.address, namespace=args.namespace, log_to_driver=False)
# ----------------------------------------------------------------------------------------
serve.start(detached=True)
# ----------------------------------------------------------------------------------------
# Gestión de CAMARAS WEB------------------------------------------------------------------
@serve.deployment(name=args.id,num_replicas=1,route_prefix=f"/status_{args.id}")
class MANAGER:
    def __init__(self, source= args.source, idc=args.id, urlservices= args.urlservices, params=params, vparams=vparams):
        # QUEUE de imagen
        self.IMGDRAW   = Queue(maxsize=1)
        self.IMGQUEUE  = Queue(maxsize=5)
        self.STREAMING = Queue(maxsize=5)
        # DIMENSION
        self.height = 0
        self.width  = 0
        # CIFRADO de link de streaming
        source = base64.b64decode(source)
        source = source.decode('ascii')
        source = str(source)
        # PARAMETROS 
        self.cap = cv2.VideoCapture(source)
        self.timeml = args.timeml
        self.timeframe = round(1/args.frame,ndigits=2)
        self.proresolucion = args.resolution
        # STATUS
        self.CAMSTATUS = False
        self.SERSTATUS = False
        self.CALSTATUS = False
        self.CALIB     = False
        self.AUXIMAGE  = False
        self.flagaux   = False
        # THREADING
        thr.Thread(name=NCAMRA + idc, target=self.UPDATEFRAME,
                   args=(idc,), daemon=True).start()
        thr.Thread(name=NQUERY + idc, target=self.ANALYTICS,
                   args=(idc, urlservices, params,), 
                   daemon=True).start()
        urlservicesview = 'http://127.0.0.1:5000/DRAWPROPERTIES'
        thr.Thread(name=TTESTY + idc, target=self.DRAWIMG,
                   args=(idc, urlservicesview, vparams,),
                   daemon=True).start()
        '''thr.Thread(name='rstp', target=self.RTSP, args=(), 
                   daemon=True).start()'''
        # EVENTOS de threading
        self.kill = thr.Event()
    # CONTROL servcio WEB -----------------------------------------------------------------------
    # Status
    def STATUS(self):
        return {"CAMERA":inf, "STATUS_CAMERA":self.CAMSTATUS, "STATUS_SERVER_ML": self.SERSTATUS, 
        "STATUS_TEST": self.CALSTATUS}
    # Habilitar calibración
    def ENABLE_CALIBRATE(self, params):
        if params == '1':
            self.CALIB = True
        else:
            self.CALIB = False
        return self.CALIB
    # __del__ 
    def __del__(self): 
        self.kill.is_set()
    # THREAD Camaras
    def UPDATEFRAME(self, idc=''):
        time.sleep(1)
        while True:
            if self.cap.isOpened():
                _, frame = self.cap.read()
                frame = imutils.resize(frame, height=self.proresolucion)
                self.Height, self.Width, _ = np.shape(frame)
                if not self.IMGDRAW.full():
                    self.IMGDRAW.put(frame)
                if not self.IMGQUEUE.full():
                    self.IMGQUEUE.put(frame)
                '''if not self.STREAMING.full():
                     self.STREAMING.put(frame)'''
                self.CAMSTATUS = True
            else:
                self.CAMSTATUS = False              
            #DELAY
            time.sleep(self.timeframe)
    # THERAD calibracion
    def DRAWIMG(self, idc, url, params):
        count = 0
        C = SERVICEML.remote()
        while True:
            count += 1
            if not self.IMGDRAW.empty():
                if self.CALIB == True:
                    try:                
                        imgqdw = self.IMGDRAW.get()
                        jsonannot, status =  ray.get(C.VWSYNC.remote(imgqdw, url, params))
                        imgstreaming = self.DRAW(imgqdw, jsonannot)
                        if status == True:
                            if not self.STREAMING.full():
                                self.STREAMING.put(imgstreaming)                   
                            #fgt = Image.fromarray(imgstreaming)
                            #fgt.save(f'/home/ossun/sfis/static/test/test{count}.jpg')
                        self.CALSTATUS = status
                    except:
                        self.CALSTATUS = False
                else:
                    self.CALSTATUS = False
            # Tiempo de ejecución
            # time.sleep(self.timeframe)
    # DIBUJO de caracteristicas de rostro, ubicacion y angulo
    def DRAW(self, imgq, annot):
    # JSON
        pts = []
        for crt in annot:
            px1, py1, px2, py2 = crt['facial_area']
            pointface = crt['landmarks']
            cv2.rectangle(imgq, (px1, py1), (px2, py2), (0, 0, 255), 2, cv2.LINE_AA)
            points = []
            for n  in pointface:
                point = crt['landmarks'][f'{n}']
                cv2.circle(imgq, point, 1, (0,255,0), -1)
                points.append(point)
            points = np.array(points)  

            pts = []
            tri = Delaunay(points)
            tri = points[tri.simplices]            
            for k in tri:
                x = sum(k[:,0])/3
                y = sum(k[:,1])/3
                pts.append([x,y])

            pointarray = np.append(points, np.array(pts), axis=0)
            triaument  = Delaunay(pointarray)
            triaument  = pointarray[triaument.simplices]
            triaument  = triaument.astype(int)
            for l in triaument:
                cv2.polylines(imgq, [l], True, (0, 255, 0), 1)
           
            Wpx, Hpx = px2 - px1, py2 - py1
            PXs = f'D: {Wpx}x{Hpx}'
            # Angulos
            Or = crt['angles']['Angles']
            (Ya, Xa, Za) = Or['Y'],Or['X'],Or['Z']
            Ostx = f'X: {Xa}'
            Osty = f'Y: {Ya}'
            #-------------------------
            cv2.putText(imgq, PXs,  (px2 + 5, py1 + 12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA )
            cv2.putText(imgq, Ostx, (px2 + 5, py1 + 28), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA )
            cv2.putText(imgq, Osty, (px2 + 5, py1 + 42), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA )
        # Caracteristicas
        return imgq
    # STREAMING
    def STREAM(self): 
        if not self.STREAMING.empty():                
            S = cv2.imencode(".jpg", self.STREAMING.get())[1].tobytes()
            self.AUXIMAGE = S
            self.flagaux = True
            return S
        else:
            if self.flagaux == True:
                return self.AUXIMAGE
            else:
                video = np.zeros([200, 300,3],np.uint8)
                S = cv2.imencode(".jpg", video)[1].tobytes()
                return S
    # THREAD identificación
    def ANALYTICS(self, idc, urlservices, params):
        time.sleep(1)       
        C = SERVICEML.remote()
        while True:
            try:
                if not self.IMGQUEUE.empty():
                    img = self.IMGQUEUE.get()
                    _ , status = ray.get(C.SASYNC.remote(img, urlservices, params))
                    self.SERSTATUS =  status
                    #DELAY
                    time.sleep(self.timeml)
            except:
                self.SERSTATUS = False
            #DELAY
            time.sleep(0.01)
# PETICION Asincrona ------------------------------------------------------------
@ray.remote
class SERVICEML:
    def __init__(self):

        self.C = CLIENTE.remote()
        self.V = CLIENTE.remote()

    def SASYNC(self, img, urlservices, params):
        DATA = []
        STATUS = False
        # Multiparte
        img = cv2.imencode('.jpg', img)[1].tobytes()
        datamulipart = aiohttp.FormData()
        datamulipart.add_field('image', img, filename='image.jpg', content_type='image/jpg')
        datamulipart.add_field('annotations', params, filename='annotations.json', content_type='application/json')
        # Task
        task = self.C.CLIENTML.remote(urlservices, datamulipart)
        # Query - asyncio
        try:        
            loop = asyncio.new_event_loop()                 
            DATA = loop.run_until_complete(task)
            STATUS = True
        except:
            STATUS = False
        finally:
            loop.close()
        return DATA, STATUS
    
    def VWSYNC(self, img, urlservices, params):
        DATA = []
        STATUS = False
        # Multiparte
        img = cv2.imencode('.jpg', img)[1].tobytes()
        datamulipart = aiohttp.FormData()
        datamulipart.add_field('image', img, filename='image.jpg', content_type='image/jpg')
        datamulipart.add_field('annotations', params, filename='annotations.json', content_type='application/json')
        # Task
        task = self.V.CLIENTML.remote(urlservices, datamulipart)
        # Query - asyncio
        try:        
            loop = asyncio.new_event_loop()                 
            DATA = loop.run_until_complete(task)
            STATUS = True
        except:
            STATUS = False
        finally:
            loop.close()
        return DATA, STATUS
    
# RESQUEST Asincrona ---------------------------------------------------------------------
@ray.remote
class CLIENTE():
    async def CLIENTML(self, urls, files):
        connector = aiohttp.TCPConnector(limit= None)  
        async with aiohttp.ClientSession(connector=connector) as client:
            async with client.post(url=urls, data=files) as answer:
                assert answer.status == 200
                JSONClient = await answer.json()
                return JSONClient
# Deploy servicio CAMARA-----------------------------------------------------------------
if args.add == True:
    MANAGER.deploy()    
if args.remove == True:
    MANAGER.delete()    
# ----------------------------------------------------------------------------------------
# SERVICIO WEB----------------------------------------------------------------------------
@serve.deployment(name=f'service_{args.id}', route_prefix=f"/WEB_{args.id}")
@serve.ingress(app)
class SERVICES_WEB():

    services = MANAGER.get_handle()
    framme = None

    @app.get("/status")
    async def status(self):
        flag = ray.get(self.services.STATUS.remote())
        return {"Information":flag}

    @app.get("/calibration/{params}")
    async def calibracion(self, params: str):
        flag = ray.get(self.services.ENABLE_CALIBRATE.remote(params))
        return {"Status_View":flag}

    def updatestream(self):       
        frame  = ray.get(self.services.STREAM.remote())
        return frame 

    @app.get("/video_feed", response_class=HTMLResponse)
    async def streaming(self):
        return Response(self.updatestream(), media_type='image/png')

# Deploy servicio WEB---------------------------------------------------------------------
if args.add == True:
    SERVICES_WEB.deploy()
if args.remove == True:
    SERVICES_WEB.delete()
# ----------------------------------------------------------------------------------------
'''
1.init cluster: ray start --head
2.init Serve:   serve start
Detener:        ray stop --force
remove:        .delete()
replica:       .deploy()
'''
