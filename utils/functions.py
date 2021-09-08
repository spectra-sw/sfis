import base64
import cv2
import numpy as np
from sqlalchemy import text
from datetime import datetime
from faker import Faker
import random
import uuid
def fakerRegistros(registros):
    #global registros
    fake = Faker()
    #cant = len(registros)
    fotos = ["5a","5b","5c","7a","7b","7c"]
    if len(registros) >20:
        registros.clear()

    for _ in range(0,10):
        now = datetime.now()
        autorizado =random.randint(0,1)
        foto=random.choice(fotos)
        if autorizado == 0:
            clase = "table-danger"
        if autorizado == 1:
            clase = "table-success"
        registro={
            "title_date": now.strftime("%d/%m/%Y %H:%M:%S"),
            'title_name': fake.name(),
            'title_idcam': str(uuid.uuid4())[:8],
            'title_authorization' : autorizado,
            "clase": clase,
            "foto": foto,
            'title_score': 0.82
        }
    
        registros.append(registro)
    return registros

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def maxId(table,db):
    
    queryId ="select max(id) as max from " + table
    sql = text(queryId)
    result = db.engine.execute(sql)
  
    id= [row for row in result]
    
    if isinstance(id[0][0], int):
        return int(id[0][0])+1
    else:
        return 1