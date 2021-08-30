from sqlalchemy import text
from flask_sqlalchemy import SQLAlchemy

def getZonas(db):
    query ="select * from zonas order by nombre asc"
    sql = text(query)
    result = db.engine.execute(sql)
    zonas=[]
    print(result)
    for row in result:
        zonas.append(row)

    return zonas     

def getZona(db,id):
    query ="select nombre from zonas where id ="+str(id)
    sql = text(query)
    result = db.engine.execute(sql)
    nombre=""
    for row in result:
        nombre = row[0]
    print(nombre)
    return nombre     