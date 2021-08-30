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