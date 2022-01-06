from elasticsearch import Elasticsearch
es = Elasticsearch()

es.indices.delete(index='aggs', ignore=[400, 404])