import elasticsearch as ES
es = ES.Elasticsearch([{'host': 'localhost', 'port': '9200'}])
arrayencoding = 512 #128
# Crear un INDEX -> Tabla en la DB
mapping = {
    "mappings": {
        "properties": {
            "title_date":{
                "type": "date",
            },
            "title_vector": {
                "type": "dense_vector",
                "dims": arrayencoding
            },
            "title_name": {
                "type": "keyword"
            },
            "title_identy": {
                "type": "integer"
            },
            "title_authorization":{
                "type": "boolean"
            }
        }
    }
}
index = "facencoding"
es.indices.create(index="facencoding", body=mapping)
