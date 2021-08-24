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