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

nest_asyncio.apply()

def getCamaraById(id,db):
    query ="select * from camaras where id ="+id
    sql = text(query)
    result = db.engine.execute(sql)
    camara=[]
    #print(result)
    for row in result:
        camara.append(row)

    #print(camara)
    return camara    

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

def activar(server,id,db,parametros,accion):
    parametros=parametros.split("&")
    dictp={}
    for p in parametros:
        d = p.split("=")
        dictp[d[0]]=float(d[1])
    
    print(dictp)
    print(accion)
    if accion =="1":
        action ="add"
    if accion == "0":
        action="remove"

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
    params = {"idc": camara[0][11],"timeml":dictp["timeml"], "frame": camara[0][4] ,"thr": dictp["thr"], "sizeface": dictp["sizeface"], "indexread": args.indexread,
            "indexwrite": args.indexwrite, "host": args.hostname, "port": args.port,
            "sizeread": dictp["sizeread"], "thrperson": dictp["thrperson"], "namespace": args.namespace,
            "nreplica":dictp["nreplica"],
            "thrminperson": dictp["thrminperson"], "cam": camara[0][8], 'uuid': camara[0][11],
            "action" : action
            }
    print(params)
    params = json.dumps(params).encode('utf-8')

    async def querysyncro(urls, files): 
        connector = aiohttp.TCPConnector(limit= None)    
        async with aiohttp.ClientSession(connector= connector) as session:        
            async with session.post(url= urls, data= files) as Respuesta:
                #print(await Respuesta.text()) 
                return await Respuesta.json()

  
    url = server + 'HARDWARE'
    
    datas = FormData()
    datas.add_field('annotations', params, filename='annotations.json', content_type='application/json')
    #SEND_> Solicitud Asincrona.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = querysyncro(url, datas)
    resp = loop.run_until_complete(task)
    #print(resp)
    return resp  

    


