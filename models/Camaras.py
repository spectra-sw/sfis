from sqlalchemy import text


def getCamaras(db):
    query ="select * from camaras"
    sql = text(query)
    result = db.engine.execute(sql)
    camaras=[]
    print(result)
    for row in result:
        camaras.append(row)

    return camaras     