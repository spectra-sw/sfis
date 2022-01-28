from sqlalchemy import text
import requests
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
    DATA = []
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
    
    cam = camara[0][8]
    cam = cam.encode('ascii')
    cam = base64.b64encode(cam)
    cam = cam.decode('ascii')
    source = cam
    
    idc = camara[0][11]
    frame = camara[0][4]
    urlservices = "http://127.0.0.1:5000/FACE_INTEGRATION"
    timeml = dictp["timeml"]
    indexread = "facencoding"
    indexwrite = "activity"
    namespace = "serve"
    nreplica = 1
    address = "auto"
    hostname = "localhost"
    port = 9200
    sizeread = 1
    sizeface = int(dictp["sizeface"])
    thr = dictp["thr"]
    thrperson = dictp["thrperson"]
    thrminperson = dictp["thrminperson"]
    thrcw = dictp["thrcw"]
    thrch = dictp["thrch"]
    action = action

    params = {"idc":idc, "source": source, "frame": frame, "urlservices": urlservices, "timeml": timeml, "indexread": indexread, "indexwrite": indexwrite, "namespace": namespace, "nreplica": nreplica, "address": address, "hostname": hostname, "port": port, "sizeread": sizeread, "sizeface": sizeface, "thr": thr, "thrperson": thrperson,  "thrminperson": thrminperson, "thrcw": thrcw, "thrch": thrch, "resolution": 720}
    # json -----------------------------------------------------------------------------------
    urladd = "http://localhost:7543/camservice/active"
    urlrem = "http://localhost:7543/camservice/deactive"
    print(params)

    try:
        if  "add" in action:
            resp = requests.post(url=urladd, json=params)
            #  print(resp.text)
        elif "remove" in action:
            resp = requests.post(url=urlrem, json=params)
            #  print(resp.tex)
    except:
        DATA = 'CHECK SERVER'
    
    return DATA # resp  

    

