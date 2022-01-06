from elasticsearch import Elasticsearch
es = Elasticsearch()

es.indices.delete(index='facencoding', ignore=[400, 404])