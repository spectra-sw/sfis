from elasticsearch import Elasticsearch
es = Elasticsearch()

es.indices.delete(index='activity', ignore=[400, 404])