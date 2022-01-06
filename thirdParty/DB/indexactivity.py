import elasticsearch as ES
es = ES.Elasticsearch([{'host': 'localhost', 'port': '9200'}])
# Crear un INDEX -> Tabla en la DB
mapping = {
    "mappings": {
        "properties": {
            "title_date":{
                "type": "date"
            },
            "title_idcam": {
                "type": "text",
                "fielddata": true 
            },
            "title_name": {
                "type": "text"
            },
            "title_identy": {
                "type": "integer"
            },
            "title_score": {
                "type": "float"
            },
            "title_authorization": {
                "type": "boolean"
            },
            "title_imagen_uuid": {
                "type": "text"
            },
            "title_face_uuid": {
                "type": "text"
            },
            "title_box_face": {
                "type": "dense_vector",
                "dims": 4
            },
        }
    }
}
indexwrite = "activity"
es.indices.create(index="activity", body=mapping)
