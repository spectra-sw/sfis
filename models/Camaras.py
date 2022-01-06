from sqlalchemy import text
import argparse
import json
from flask import render_template, request
import asyncio
import aiohttp
import nest_asyncio
from aiohttp import FormData
import json
from utils.functions import maxId
import shortuuid
import base64
from subprocess import check_output
import cv2

nest_asyncio.apply()

def getCamaraById(id,db):
    query ="select * from camaras where id ='"+id+"'"
    sql = text(query)
    result = db.engine.execute(sql)
    camara=[]
    #print(result)
    for row in result:
        camara.append(row)

    #print(camara)
    return camara    

def getInfCamara(uuid,db):
    query ="select id_zona from camaras where uuid ='"+uuid+"'"
    sql = text(query)
    result = db.engine.execute(sql)
    Camara = [] 
    #print(result)
    for row in result:
        Camara.append(row[0]) 
    return Camara

def getZonaCamara(uuid,db):
    query ="select nombre from camaras where uuid ='"+uuid+"'"
    sql = text(query)
    result = db.engine.execute(sql)
    zona=""
    #print(result)
    for row in result:
        zona=row[0]
    #print(query)
    #print(zona+ " " + uuid)
    return zona    

def getCamaras(db):
    query ="select * from camaras"
    sql = text(query)
    result = db.engine.execute(sql)
    camaras=[]
    #print(result)
    for row in result:
        camaras.append(row)

    print(camaras)
    return camaras     

def guardarCamara(datos,db):
    camara_id=maxId('camaras',db)
    uuid = shortuuid.ShortUUID().random(length=15)
    query ="insert into camaras values ( \
            "+ str(camara_id) + ", \
            '" + str(datos['nombre']) + "', \
            '" + str(datos['ref']) + "', \
            '" + str(datos['marca']) + "', \
            " + datos['fps'] +", \
            '" + datos['ip'] +"', \
            " + datos['zona'] +", \
            " + datos['puerto'] +", \
            '" + datos['link'] +"', \
            '" + datos['usuario'] +"', \
            '" + datos['password'] +"', \
            '" + uuid +"' \
            )"
    print(query)
    sql = text(query)
    result = db.engine.execute(sql)

def testcamara(id,db):
    camara = getCamaraById(id,db)
    url =camara[0][8]
    # try to open the stream
    print(url)
    cap = cv2.VideoCapture(url)
    ret = cap.isOpened()  # if it was succesfully opened, that's the URL you need
    cap.release()
    return str(ret)


def activar(passw,server,id,db,parametros,accion):
    parametros=parametros.split("&")
    dictp={}
    for p in parametros:
        d = p.split("=")
        dictp[d[0]]=float(d[1])
    
    print(dictp)
    print(accion)
    if accion =="1":
        action = "add"
    if accion == "0":
        action= "remove"

    camara = getCamaraById(id,db)
    #print(camara)
    
    parser = argparse.ArgumentParser(description="Setting")
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--source', type=None, default=None)
    parser.add_argument('--frame', type=int, default=10)
    parser.add_argument('--urlservices', type=str, default='')
    parser.add_argument('--timeml', type=float, default=1.0)
    parser.add_argument('--indexread', type=str, default='facencoding')
    parser.add_argument('--indexwrite', type=str, default='activity')
    parser.add_argument('--namespace', type=str, default='serve')
    parser.add_argument('--nreplica', type=int, default=1)
    parser.add_argument('--address', type=str, default='auto')
    parser.add_argument('--hostname', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9200)
    parser.add_argument('--sizeread', type=int, default=1)
    parser.add_argument('--sizeface', type=int, default=25)
    parser.add_argument('--thr', type=float, default=0.85)
    parser.add_argument('--thrperson', type=float, default=50.0)
    parser.add_argument('--thrminperson', type=float, default=40.0)
    parser.add_argument('--remove', type=bool, default=False)
    parser.add_argument('--add', type=bool, default=False)
    args = parser.parse_args()
    
    
    # json -----------------------------------------------------------------------------------
    params = {"idc": camara[0][11],"timeml":dictp["timeml"], "frame":int(camara[0][4]) ,"thr": dictp["thr"], "sizeface": int(dictp["sizeface"]), "indexread": args.indexread,
            "indexwrite": args.indexwrite, "host": args.hostname, "port": args.port,
            "sizeread":1, "thrperson": dictp["thrperson"], "namespace": args.namespace,
            "nreplica":1,
            "thrminperson": dictp["thrminperson"], "cam": camara[0][8], 'uuid': camara[0][11],
            "action" : action,
            "thrcw":dictp["thrcw"], "thrch":dictp["thrch"]
            }
    print(params)


    uid             = params['idc']
    # CODE link Streaming
    cam = params['cam']
    cam = cam.encode('ascii')
    cam = base64.b64encode(cam)
    cam = cam.decode('ascii')
    # -------------------
    source          = cam
    frame           = params['frame']
    urlservices     = "http://127.0.0.1:5000/FACE_INTEGRATION"
    timeml          = params['timeml']
    indexread       = params['indexread']
    indexwrite      = params['indexwrite']
    namespace       = params['namespace']
    nreplica        = params['nreplica']
    address         = "auto"
    hostname        = "localhost"
    port            = params['port']
    sizeread        = params['sizeread']
    sizeface        = params['sizeface']
    thr             = params['thr']
    thrperson       = params['thrperson']
    thrminperson    = params['thrminperson']
    action          = params['action']
    
    try:
        if  "add" in action:            
            commandadd = 'echo '+passw+' | sudo -S python3 ../Serveargument.py '+' --id '+uid+' --source '+source+' --frame '+str(frame)+' --urlservices '+urlservices+' --timeml '+str(timeml)+' --indexread '+indexread+' --indexwrite '+indexwrite+' --namespace '+namespace+' --nreplica '+str(nreplica)+' --address '+address+' --hostname '+hostname+' --port '+str(port)+' --sizeread '+str(sizeread)+' --sizeface '+str(sizeface)+' --thr '+str(thr)+' --thrperson '+str(thrperson)+' --thrminperson '+str(thrminperson)+' --add '+str(True)
            DATA = check_output(commandadd, shell=True).decode('utf-8')
        elif "remove" in action:
            commandadd = 'echo '+passw+' | sudo -S python3 ../Serveargument.py '+' --id '+uid+' --source '+source+' --frame '+str(frame)+' --urlservices '+urlservices+' --timeml '+str(timeml)+' --indexread '+indexread+' --indexwrite '+indexwrite+' --namespace '+namespace+' --nreplica '+str(nreplica)+' --address '+address+' --hostname '+hostname+' --port '+str(port)+' --sizeread '+str(sizeread)+' --sizeface '+str(sizeface)+' --thr '+str(thr)+' --thrperson '+str(thrperson)+' --thrminperson '+str(thrminperson)+' --remove '+str(True)
            DATA = check_output(commandadd, shell=True).decode('utf-8')
    except:
        DATA = 'CHECK SERVER'
    
    print("DATA: ", DATA)
    return DATA#resp  

    

