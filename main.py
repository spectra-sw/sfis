from flask_cors import CORS
from flask import Flask
from flask import render_template, request, redirect, url_for
from flask import   make_response, Response, jsonify
import cv2
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from sqlalchemy import create_engine
import base64
from imagecapture import capturar
import numpy as np
#from api import *
import random
from datetime import datetime
from faker import Faker
import uuid
import aiohttp
import asyncio
import nest_asyncio
from subprocess import check_output
from datetime import datetime, timedelta
from threading import Thread
nest_asyncio.apply()


from models.Zonas import *
from models.Visitantes import *
from models.Camaras import *
from models.Ingresos import *

from utils.functions import *

app = Flask(__name__)
app.config.from_pyfile('config.py')
server=app.config['SERVERML']
"""
INCIAL ESTRUCTURES
"""

registros=[]
registros2=[]
db = SQLAlchemy(app)


"""
ROUTES
"""

@app.route('/')
def inicio():
    return render_template('login.html', titulo="SFIS")

@app.route('/inicio',methods=['GET','POST'])
def minicio():
    return render_template('inicio.html', titulo="SFIS")

@app.route('/menu',methods=['GET','POST'])
def menu():
    return render_template('menu.html', titulo="SFIS")

@app.route('/ingreso')
def ingreso2():
    zonas = getZonas(db)
    return render_template('ingreso.html', titulo="SFIS - Ingreso visitantes", zonas=zonas)

@app.route('/capturar')
def foto():
    capturar()

@app.route('/registro',methods=['POST'])
def registro():
    if request.method == 'POST':
        print("POST")
        print("Registrando usuario")
        nombre= request.form['nombre']
        tipoid = request.form['tipoid']
        id = request.form['id']
        empresa = request.form['empresa']
        correo = request.form['correo']
        telefono = request.form['telefono']
        
        
        visitante_id=maxId('visitantes',db)
        query ="insert into visitantes values ("+ str(visitante_id) + ",'" + str(tipoid) + "','" +  str(id) + "','" + str(nombre) + "','" + str(correo) +"','" + str(empresa) +"','" + str(telefono) +"')"
        #return query
        
        sql = text(query)
        result = db.engine.execute(sql)

        fphoto  = request.form['fphoto']
        img = readb64(fphoto)
        filef = "static/users/" + str(visitante_id) + "a.jpg"
        cv2.imwrite(filef,img)

        fphoto2  = request.form['fphoto2']
        img = readb64(fphoto2)
        fileli = "static/users/" + str(visitante_id) + "b.jpg"
        cv2.imwrite(fileli,img)

        fphoto3  = request.form['fphoto3']
        img = readb64(fphoto3)
        fileld = "static/users/" + str(visitante_id) + "c.jpg"

        cv2.imwrite(fileld,img)

        datos={
            'nombre' : nombre,
            'cc' :id,
            'imgf' : filef,
            'imgld' : fileld,
            'imgli' : fileli
        }
        datosIngreso={
            'id_visitante':visitante_id,
            'destino' : request.form['destino'],
            'contacto' : request.form['contacto'],
            'motivo' : request.form['motivo'],
        }
        
        registroIngreso(datosIngreso,db)
        registroRemoto(datos)
        return render_template('menu.html', titulo="SFIS")
        #return jsonify("Visitante creado")


@app.route('/monitor')
def monitor():
    return render_template('monitor2.html', titulo="SFIS")

@app.route('/actividad')
def actividad():
    camaras = getCamaras(db)
    print(camaras)
    
    commandadd = 'sudo cp -Ru /var/lib/docker/volumes/activity/_data/. static/activity/'
    DATA = check_output(commandadd, shell=True).decode('utf-8')
    commandadd = 'sudo cp -Ru /var/lib/docker/volumes/environment/_data/. static/environment/'
    DATA = check_output(commandadd, shell=True).decode('utf-8')
    
    #Thread(target=copyImages,daemon=True).start()

    return render_template('actividad.html', titulo="SFIS",camaras=camaras)

@app.route('/rutas')
def rutas():
    zonas = getZonas(db)
    return render_template('rutas.html', titulo="SFIS", zonas=zonas)

@app.route('/activity/get/<id>',methods=['GET'])
def activityd(id):
    global registros
    return render_template('cardactivity.html', dato=registros[int(id)])

@app.route('/activity/get2/<id>',methods=['GET'])
def activityd2(id):
    global registros2
    return render_template('cardactivity2.html', dato=registros2[int(id)])

@app.route('/activity/get',methods=['GET'])
def activity():
    global registros
    global min
     
    #registros = fakerRegistros(registros)
    
    registros=pruebafiltro()
    #print(registros)
    for r in registros:
        if r['title_authorization']==False:
            r['clase'] = "table-danger"
        if r['title_authorization']==True:
            r['clase'] = "table-success"

        zona = getZonaCamara(r['title_idcam'],db)
        r['zona'] = zona
        r['foto'] = r['title_face_uuid']
        r['env'] = r['title_imagen_uuid']
        #print(r['foto'])
        '''
        filePhoto= 'static/activity/'+ r['title_uuid']+'.jpg'
        r['foto'] = r['title_uuid']+'.jpg'
        #jpg_original = base64.b64decode(r['title_imagen'])
        
        
        with open(filePhoto, 'wb') as f_output:
            f_output.write(jpg_original)
        '''
    #print(registros)
    
    commandadd = 'sudo cp -Ru /var/lib/docker/volumes/activity/_data/. static/activity/'
    DATA = check_output(commandadd, shell=True).decode('utf-8')
    commandadd = 'sudo cp -Ru /var/lib/docker/volumes/environment/_data/. static/environment/'
    DATA = check_output(commandadd, shell=True).decode('utf-8')
    print("Imagenes copiadas")
    
    #Thread(target=copyImages,daemon=True).start()
    return render_template('registros.html', registros= registros)



@app.route('/faces/get',methods=['GET'])
def faces():
    print(len(registros))
    return render_template('faces.html', faces= registros) 


@app.route('/ingresos',methods=['GET','POST'])
def ingresos():
    query ="select * from ingresos order by fecha asc"
    sql = text(query)
    result = db.engine.execute(sql)
    ingresos=[]
    print(result)
    for row in result:
        ingresos.append(list(row))
        ingresos[len(ingresos)-1][1]=getNombre(db,row[1])
        ingresos[len(ingresos)-1][3]=getZona(db,row[3])
        foto = str(row[1])+"a.jpg"
        ingresos[len(ingresos)-1].append(foto)
    print(ingresos)
    '''
    for i in ingresos:
        nombre =getNombre(db,i[1])
        i[1]=getNombre(db,nombre)
    '''

    return render_template('ingresos.html', titulo="SFIS", ingresos=ingresos)
@app.route('/personas',methods=['GET','POST'])
def personas():
    query ="select * from visitantes order by id asc"
    sql = text(query)
    result = db.engine.execute(sql)
    visitantes=[]
    print(result)
    for row in result:
        visitantes.append(list(row))
        foto = str(row[0])+"a.jpg"
        visitantes[len(visitantes)-1].append(foto)
    print(visitantes)
    '''
    for i in ingresos:
        nombre =getNombre(db,i[1])
        i[1]=getNombre(db,nombre)
    '''

    return render_template('personas.html', titulo="SFIS", visitantes=visitantes)

@app.route('/zonas',methods=['GET','POST'])
def zonas():
    zonas = getZonas(db)
    return render_template('zonas.html', titulo="SFIS", zonas=zonas)

@app.route('/camaras',methods=['GET','POST'])
def camaras():
    camaras = getCamaras(db)
    zonas= getZonas(db)
    return render_template('camaras.html', titulo="SFIS", camaras=camaras,zonas=zonas)



@app.route('/pruebaregistro',methods=['GET','POST'])
def pruebaregistro():
    registroRemoto()
    return "Registro creado"

@app.route('/buscarp',methods=['POST'])
def buscarp():
    if request.method == 'POST':
        data = request.form.to_dict()
        tipoid= request.form['tipoid']
        id= request.form['id']
        datos=getDatosVisitante(tipoid,id,db)
        print(datos)
        return  render_template('datosp.html',  datos=datos)

@app.route('/activarcamara',methods=['POST'])
def activarcamara():  
    if request.method == 'POST':
        id= request.form['id']  
        parametros=request.form['parametros'] 
        accion =request.form['accion'] 
        #print(parametros) 
        data=activar(server,id,db,parametros,accion)    

        return str(data)    
@app.route('/testcamara',methods=['POST'])
def tcamara():  
    if request.method == 'POST':
        id= request.form['id']  
        res=testcamara(id,db)

        return res   

@app.route('/guardarcamara',methods=['POST'])
def gcamara():

    if request.method == 'POST':
        data = request.form.to_dict()
        print(data)
        guardarCamara(data,db)
    
    return 'Cámara creada'
    

def registroRemoto(datos):
    async def querysyncro(urls, files): 
        connector = aiohttp.TCPConnector(limit= None)    
        async with aiohttp.ClientSession(connector= connector) as session:        
            async with session.post(url= urls, data= files) as Respuesta:
                #print(await Respuesta.text()) 
                return await Respuesta.json()
                
    from aiohttp import FormData
    import json
    urlregitro = server + 'REGISTRO'
    #JSON _> Parametros.
    nombre = datos['nombre']
    cc = datos['cc']
    access = True
    host = "localhost"
    port = 9200
    indexwrite = "facencoding"
    imgf =  datos['imgf']
    imgld = datos['imgld']
    imgli = datos['imgli']
    IMG = [imgf, imgld, imgli]
    idc = "XXXXX"
    config = { "idc": idc , "thr": 0.85, "size": 12, "name": nombre, "CC": cc, "Access": access, "host": host,
    "port": port, "indexwrite": indexwrite }
    json_config = json.dumps(config).encode('utf-8')
    #FRAME_> OpenCV.
    for urlimg in IMG:
        FRAME = cv2.imread(urlimg) # Remplazar por FRAME DEl VIDEO
        img = cv2.imencode('.jpg', FRAME)[1].tobytes()
        #DATA_> Estructura de datos.
        datas = FormData()
        datas.add_field('image', img, filename='image.jpg', content_type= 'image/jpg')
        datas.add_field('annotations', json_config, filename='annotations.json', content_type='application/json')
        #SEND_> Solicitud Asincrona.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = querysyncro(urlregitro, datas)
        resp = loop.run_until_complete(task)
        assert resp["_shards"]["successful"] == 1
    


@app.route('/server',methods=['POST'])
def initserver():
    data = request.form.to_dict()
    status = data['status']
    print(status)
    if status=="2":
        try:           
            command = 'bentoml serve-gunicorn FaceOnnx:latest'
            check_output(args=command, shell=True).decode('utf-8')
            DATA = 'SERVER ML STARTED'
        except:
                DATA = 'SERVER ML RUNNING'

    else:
        if status == "1":
            config = {"STATUS": 'START'}
        if status == "0":
            config = {"STATUS": 'STOP'}

        if 'START' in config['STATUS']:
            try:           
                command = 'ray start --head && serve start'
                check_output(args=command, shell=True).decode('utf-8')
                DATA = 'SERVER STARTED'
            except:
                DATA = 'SERVER RUNNING'
        elif 'STOP' in config['STATUS']:
            command = 'ray stop --force'
            check_output(args=command, shell=True).decode('utf-8')
            DATA = 'SERVER STOPPED'

    return DATA  


@app.route('/pruebafiltro',methods=['GET','POST'])
def pruebafiltro():
    
    async def querysyncro(urls, files): 
        connector = aiohttp.TCPConnector(limit= None)    
        async with aiohttp.ClientSession(connector= connector) as session:        
            async with session.post(url= urls, data= files) as Respuesta:
                #print(await Respuesta.text()) 
                return await Respuesta.json()

    from aiohttp import FormData
    import json
    urlregitro = server + 'READFILTRO'
    #JSON _> Parametros.1
    host = "localhost"
    port = 9200
    indexread = "activity"
    # Las fechas deben contener al Z al final para la zona horaria
    # Si se usa "now" es el limite actual
    #fmin = datetime.today() - timedelta(hours=0, minutes=5)
    minutes = timedelta(minutes=10)
    now = datetime.now()
    fmin =(now-minutes).strftime("%Y-%m-%d %H:%M:%S")
    fmin = fmin[0:10]+"T"+fmin[11:]+"Z"
    print("Min"+fmin)
    #fmin = "2021-08-27T14:30:10Z" 

    #fmin=fecha+"T"+str(hora)+":"+str(min)+":"+str(seg)+"Z"
    #print(fmin)
    #fmin ="now-30m"
    #fmax = "now"
    
    now = datetime.now()
    fmax =now.strftime("%Y-%m-%d %H:%M:%S")
    fmax = fmax[0:10]+"T"+fmax[11:]+"Z"
    print("Max"+fmax)
    
    
    config = {"host": host, "port": port, "indexread": indexread, "fmin": fmin, "fmax": fmax ,"sizedataread": 20, "search":"FECHA"}
    json_config = json.dumps(config).encode('utf-8')
    #FRAME_> OpenCV.

    #DATA_> Estructura de datos.
    datas = FormData()
    datas.add_field('annotations', json_config, filename='annotations.json', content_type='application/json')
    #SEND_> Solicitud Asincrona.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = querysyncro(urlregitro, datas)
    resp = loop.run_until_complete(task)
    #print(resp)
    return resp  
    
    '''

    return registros

    '''
    
@app.route('/buscaract',methods=['GET','POST'])
def buscaract():
    global registros2
    data = request.form.to_dict()
    print(data)
    

    async def querysyncro(urls, files): 
        connector = aiohttp.TCPConnector(limit= None)    
        async with aiohttp.ClientSession(connector= connector) as session:        
            async with session.post(url= urls, data= files) as Respuesta:
                #print(await Respuesta.text()) 
                return await Respuesta.json()

    from aiohttp import FormData
    import json
    urlregitro = server + 'READFILTRO'
    host = "localhost"
    port = 9200
    indexread = "activity"
   
    minutes = timedelta(minutes=10)
    now = datetime.now()
    fmin =data['desde']
    fmin = fmin+"Z"
    print("Min"+fmin)
    
    now = datetime.now()
    fmax =data['hasta']
    fmax = fmax+"Z"
    print("Max"+fmax)

    search="FECHA"
    if data['cc'] !='':
        CC = data['cc']
        search = "FECHA&CC"
    
    if data['cam'] !='':
        search="CAMDATE"
        if data['cc'] !='':
            search = "CAMDATEDOC"
    
    
    
    config = {"host": host, "port": port, "indexread": indexread, "fmin": fmin, "fmax": fmax ,"sizedataread": 1000, "search":search , 'CC':data['cc'], 'cam':data['cam']}
    json_config = json.dumps(config).encode('utf-8')
    #FRAME_> OpenCV.

    #DATA_> Estructura de datos.
    datas = FormData()
    datas.add_field('annotations', json_config, filename='annotations.json', content_type='application/json')
    #SEND_> Solicitud Asincrona.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = querysyncro(urlregitro, datas)
    resp = loop.run_until_complete(task)
    print(resp)
    #return resp 
    registros2 = resp

    for r in registros2:
        if r['title_authorization']==False:
            r['clase'] = "table-danger"
        if r['title_authorization']==True:
            r['clase'] = "table-success"
        
        r['foto'] = r['title_face_uuid']
        r['env'] = r['title_imagen_uuid']
        zona = getZonaCamara(r['title_idcam'],db)
        r['zona'] = zona

    return render_template('registros2.html', registros= registros2) 

    
    

if __name__=="__main__":
    print(app.config)
    commandadd = 'sudo cp -Ru /var/lib/docker/volumes/activity/_data/. static/activity/'
    DATA = check_output(commandadd, shell=True).decode('utf-8')
    commandadd = 'sudo cp -Ru /var/lib/docker/volumes/environment/_data/. static/environment/'
    DATA = check_output(commandadd, shell=True).decode('utf-8')
    #Thread(target=copyImages,daemon=True).start()

    app.run(host=app.config['HOST'],port=app.config['PORT'])
   
    
    
