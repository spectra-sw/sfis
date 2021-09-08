from sqlalchemy import text


def getNombre(db,id):
    query ="select nombre from visitantes where id ="+str(id)
    sql = text(query)
    result = db.engine.execute(sql)
    nombre=""
    for row in result:
        nombre = row[0]
    print(nombre)
    return nombre     

def getDatosVisitante(tipoid,id,db):
    query ="select * from visitantes where tipo_id ='"+ tipoid + "' and no_identificacion ='"+ id +"' limit 1"
    sql = text(query)
    result = db.engine.execute(sql)
    
    for row in result:
        datos=row
    
    return datos