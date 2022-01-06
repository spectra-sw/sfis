from sqlalchemy import text
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from utils.functions import *

def registroIngreso(datos,db):
    ingreso_id=maxId('ingresos',db)
    now = datetime.now()
    query ="insert into ingresos values ("+ str(ingreso_id) + "," + str(datos['id_visitante']) + ",'" +  now.strftime("%Y-%m-%d %H:%M:%S") + "','" + datos['destino'] + "','" + datos['contacto'] +"','" + datos['motivo'] +"')"
    print(query)
    sql = text(query)
    result = db.engine.execute(sql)

def getDatosIngreso(id,db):
    query ="select * from ingresos where id ='"+str(id)+"' order by fecha desc limit 1"
    sql = text(query)
    result = db.engine.execute(sql)
    
    for row in result:
        datos=row
    
    return datos
