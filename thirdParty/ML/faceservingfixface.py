from inspect import _empty
from typing import List, Sequence
from datetime import datetime as dt
from PIL import Image
import numpy as np
import uuid
import imutils
import math
import cv2
import gc
# DB Elasticsearch
import elasticsearch as ES
# Servicio de ML
import bentoml
import imageio
from bentoml.types import JsonSerializable
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact
from bentoml.adapters import ImageInput, JsonInput, DefaultOutput, AnnotatedImageInput, annotated_image_input
# MAX BATCH------------------------
MaxBatch = 32
MaxLantencia = 10000
# Ejecución GPU con onnxruntime-gpu
#onnxruntime-gpu=1.9.0
@bentoml.env( infer_pip_packages=True,pip_packages=['onnxruntime-gpu==1.9.0',
             'pillow','imageio','numpy','elasticsearch','imutils']  )
@bentoml.artifacts([OnnxModelArtifact('facesenco', backend='onnxruntime-gpu'),
                    OnnxModelArtifact('alignface', backend='onnxruntime-gpu'),
                    PickleArtifact('labels')])

class FaceOnnx(bentoml.BentoService):

    def PREPROCESSENCODE(self, input_data, dim):
        image = np.array(input_data)
        image = cv2.resize(image, (dim, dim))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    def RESIZE_IMAGE(self, img, scales):
        img_w, img_h = img.shape[0:2]
        target_size = scales[0]
        max_size = scales[1]

        if img_w > img_h:
            im_size_min, im_size_max = img_h, img_w
        else:
            im_size_min, im_size_max = img_w, img_h

        im_scale = target_size / float(im_size_min)

        if np.round(im_scale * im_size_max) > max_size:
            im_scale = max_size / float(im_size_max)

        if im_scale != 1.0:
            img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        return img, im_scale

    def ROTPOINT(self, eyes, center, theta):
        x,   y = eyes
        xo, yo = center
        xr = math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo) + xo
        yr = math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo) + yo
        return [int(xr),int(yr)]

    def FINDEUDISTANCE(self, srp, trp):
        eudistance = srp - trp
        eudistance = np.sum(np.multiply(eudistance, eudistance))
        eudistance = np.sqrt(eudistance)
        return eudistance
    
    def ANGLES(self, w, h, right_eye, left_eye, nose, mouth_right, mouth_left,):
        # punto de face
        pface = np.array([
            nose, # nose
            nose, # SolvePNP necesita 6 puntos.
            left_eye, # left eyes
            right_eye, # right eyes
            mouth_left, # Left Mouth
            mouth_right  # Right mouth
            ], dtype="double")
        # Punto de referecia 3D
        pmodel =  np.array([
            (   0.0,    0.0,    0.0), # nose
            (   0.0,    0.0,    0.0), # nose
            (-175.0,  170.0, -135.0), # left eyes
            ( 175.0,  170.0, -135.0), # right eyes
            (-150.0, -150.0, -125.0), # Left Mouth
            ( 150.0, -150.0, -125.0)  # Right mouth
        ])
        dst_focal = w                 # distancia focal (aproxima al ancho de la imagen)
        center = (w/2, h/2)           # centro de la imagen
        camera_matrix = np.array([    # matriz de camara
            [dst_focal, 0, center[0]],
            [0, dst_focal, center[1]],
            [0, 0, 1]], 
            dtype="double"
            )
        dist_coef = np.zeros((4, 1), dtype=np.float64)   # coeficienter de distorsion (se asume que: No existe distorsion)
        (success, rvector, tvector) = cv2.solvePnP(pmodel, pface, camera_matrix, dist_coef, cv2.SOLVEPNP_ITERATIVE)
        rmat, jac = cv2.Rodrigues(rvector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        Y = np.round(angles[0], 2)
        X = np.round(angles[1], 2)
        Z = np.round(angles[2], 2)
        # Pr0yeccion
        nose2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), 
                           rvector, tvector, camera_matrix, np.zeros((4,1)))       

        return Y, X, Z, nose2D[0][0][0], nose2D[0][0][1]
    
    def ALIGNMENTFACES(self,w, h, img, area_facial, right_eye, left_eye, 
                       nose, mouth_right, mouth_left, thrcw, thrch):
        rotimg = []
        properties = []
        imgandangle = []
        global point_3rd
        # Separación de cordenadas.
        left_eye_x,   left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye
        # Direccion de rotación
        try:
            if  (left_eye_y > right_eye_y):
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1 # Rotación manecillas de reloj.
            else:
                point_3rd = (left_eye_x, right_eye_y)
                direction =  1 # Rotación encontra de las manecillas de reloj.
            # Hallar los puntos del triangulo.
            a = self.FINDEUDISTANCE(np.array(left_eye),  np.array(point_3rd))
            b = self.FINDEUDISTANCE(np.array(right_eye), np.array(point_3rd))
            c = self.FINDEUDISTANCE(np.array(right_eye), np.array( left_eye))
            # Regla de coseno
            if b != 0 and c != 0: 
                cos_a = (b*b + c*c - a*a)/(2*b*c)
                angle = np.arccos(cos_a) # angulo en grados a radianes
                angle = (angle * 180) / math.pi # Radianes a grados          
                # rotacion de la imagen.                
                if   direction == -1:
                    angle = 90 - angle 
                elif direction ==  1:
                    angle = angle      

                anglerot = -1 * direction * angle
                Y, X, Z, Pxf, Pyf = self.ANGLES(
                                w=w, 
                                h=h, 
                                right_eye=right_eye, 
                                left_eye=left_eye, 
                                nose=nose, 
                                mouth_right=mouth_right, 
                                mouth_left=mouth_left
                                )

                if (X >= -1*thrcw and X <= thrcw and Y >= -1*thrch and Y <= thrch):
                    img = Image.fromarray(img)
                    rotimg = np.array(img.rotate(anglerot))
                    properties = [rotimg, {'Angles':{'Y':Y, 'X':X, 'Z':Z}, 'projectPoints':{'Pi':{'Xi':nose[0], 'Yi':nose[1]},
                                       'Pf':{'Xf':Pxf, 'Yf':Pyf }}}]
        except:
            pass
        return properties

    def ANCHORS_PLANE(self, height, width, stride, base_anchors):
        A = base_anchors.shape[0]
        c_0_2 = np.tile(np.arange(0, width)[np.newaxis, :, np.newaxis, np.newaxis], (height, 1, A, 1))
        c_1_3 = np.tile(np.arange(0, height)[:, np.newaxis, np.newaxis, np.newaxis], (1, width, A, 1))
        all_anchors = np.concatenate([c_0_2, c_1_3, c_0_2, c_1_3], axis=-1) * stride + np.tile(base_anchors[np.newaxis, np.newaxis, :, :], (height, width, 1, 1))
        return all_anchors

    def CLIP_BOXES(self, boxes, im_shape):        
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes
    
    def BBOX_PRED(self, boxes, box_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1]>4:
            pred_boxes[:,4:] = box_deltas[:,4:]
        return pred_boxes
  
    def LANDMARK_PRED(self, boxes, landmark_deltas):
        f = 1.0
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths  = boxes[:, 2] - boxes[:, 0] +  f
        heights = boxes[:, 3] - boxes[:, 1] +  f
        ctr_x = boxes[:, 0] + 0.5 * (widths  - f)
        ctr_y = boxes[:, 1] + 0.5 * (heights - f)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:,i,0] = landmark_deltas[:,i,0] * widths + ctr_x
            pred[:,i,1] = landmark_deltas[:,i,1] * heights + ctr_y
        return pred
    
    def _NMS(self, dets, threshold):
        x1 =     dets[:, 0]
        y1 =     dets[:, 1]
        x2 =     dets[:, 2]
        y2 =     dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        ndets = dets.shape[0]
        suppressed = np.zeros((ndets), dtype=np.int)

        keep = []
        for _i in range(ndets):
            i = order[_i]
            if suppressed[i] == 1:
                continue
            keep.append(i)
            ix1 = x1[i]; iy1 = y1[i]; ix2 = x2[i]; iy2 = y2[i]
            iarea = areas[i]
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                xx1 = max(ix1, x1[j]); yy1 = max(iy1, y1[j]); xx2 = min(ix2, x2[j]); yy2 = min(iy2, y2[j])
                w = max(0.0, xx2 - xx1 + 1); h = max(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= threshold:
                    suppressed[j] = 1
        return keep 

    def READES(self, params={}, vector=[]):
        MATCH = []
        table = ES.Elasticsearch(
            [{'host': params['host'], 'port': params['port']}])
        index = params['indexread']
        query = {
            "size": params['sizeread'],
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'title_vector') + 1.0",
                        "params": {
                            "queryVector": list(vector)
                        }
                    }
                }
            }
        }
        data = table.search(index=index, body=query, ignore=400)
        for i in data['hits']['hits']:
            MATCH.append({"Nombre": i['_source']['title_name'], "CC": i['_source']['title_identy'],
                          "score": round((i['_score']-1)*100, 2), "Access": i['_source']['title_authorization']})
        # MATCH: [[nombre, CC, Score, acceso], [N]]
        return MATCH
# WRITE db ElasticSearch

    def WRITEES(self, jsonlist=[], params=[], env='', face='', box_face=[]):
        es = ES.Elasticsearch([{'host': params['host'], 'port': params['port']}])
        index = params['indexwrite']
        for i in jsonlist:
            if i['score'] >= params['thrperson']:
                doc = {"title_date": dt.now(), "title_idcam": params['idc'], "title_name": i['Nombre'],
                       "title_identy": i['CC'], "title_score": float(i['score']), "title_authorization": i['Access'],
                       "title_imagen_uuid": env, "title_face_uuid": face, "title_box_face": box_face
                       }
            # elif i['score'] > i['thrminperson'] and i['score'] < params['thrperson']:
            else:
                doc = {"title_date": dt.now(), "title_idcam": params['idc'], "title_name": "DESCONOCIDO",
                       "title_identy": 0, "title_score": 0.0, "title_authorization": False,
                       "title_imagen_uuid": env, "title_face_uuid": face, "title_box_face": box_face
                       }
            # refrest id
            es.indices.refresh(index=index)
            count = es.cat.count(index=index, params={
                                 "format": "json"})[0]['count']
            # Write actividad
            es.create(index=index, id=params['idc'] + '_' + count, body=doc)
    
    def READFILTERDATE(self, emin, emax, host, port, index, size):
        DATA = []
        es = ES.Elasticsearch([{'host': host, 'port': port}])
        querys = {
            "size": size,
            "sort": {
                "title_date": "desc"
            },
            "query": {
                "range": {
                    "title_date": {
                        "gte": emin,
                        "lte": emax
                    }
                }
            }
        }
        try:
            data = es.search(index=index, body=querys)
            for i in data['hits']['hits']:      
                DATA.append(i['_source'])
        except:
            print('CHECK IF THE DATABASE IS ACTIVE')
        return DATA
    
    def READFILTERDOCUMENT(self, document, host, port, index, size):
        DATA = []
        es = ES.Elasticsearch([{'host': host, 'port': port}])
        querys = {
            "size": size,
            "sort": {
                "title_date": "desc"
            },
            "query": {
                "bool": {
                    "must": [{
                        "match": {
                            "title_identy": document
                        }
                    }],
                }
            }
        }
        try:
            data = es.search(index=index, body=querys)
            for i in data['hits']['hits']:      
                DATA.append(i['_source'])
        except:
            print('CHECK IF THE DATABASE IS ACTIVE')
        return DATA

    def READFILTERDATEDOCUMENT(self, document, emin, emax, host, port, index, size):
        DATA = []
        es = ES.Elasticsearch([{'host': host, 'port': port}])
        querys = {
            "size": size,
            "sort": {
                "title_date": "desc"
            },
            "query": {
                "bool": {
                    "must": [{
                        "match": {
                            "title_identy": document
                        }
                    }],
                    "filter": {
                        "range": {
                            "title_date": {
                                "gte": emin,
                                "lte": emax
                            }
                        }
                    }
                }
            }
        }
        try:
            data = es.search(index=index, body=querys)
            for i in data['hits']['hits']:      
                DATA.append(i['_source'])
        except:
            print('CHECK IF THE DATABASE IS ACTIVE')
        return DATA

    def READFILTERDATECAM(self, emin, emax, cam, host, port, index, size):
        DATA = []
        es = ES.Elasticsearch([{'host': host, 'port': port}])
        querys = {
            "size": size,
            "sort": {
                "title_date": "desc"
            },
            "query": {
                "bool": {
                    "must": [
                        { "match": { "title_idcam": cam }}
                    ],
                    "filter": {
                        "range": {
                            "title_date": {
                                "gte": emin,
                                "lte": emax
                            }
                        }
                    }
                }
            }
        }
        try:
            data = es.search(index=index, body=querys)
            for i in data['hits']['hits']:      
                DATA.append(i['_source'])
        except:
            print('CHECK IF THE DATABASE IS ACTIVE')
        return DATA
    
    def READFILTERDATEDOCUMENTCAM(self, document, emin, emax, cam, host, port, index, size):
        DATA = []
        es = ES.Elasticsearch([{'host': host, 'port': port}])
        querys = {
            "size": size,
            "sort": {
                "title_date": "desc"
            },
            "query": {
                "bool": {
                    "must": [
                        { "match": { "title_identy": document }},
                        { "match": { "title_idcam": cam   }}
                    ],                    
                    "filter": [
                        { "range": { "title_date": { "gte": emin, "lte": emax }}}
                    ]
                }
            }
        }
        try:
            data = es.search(index=index, body=querys)
            for i in data['hits']['hits']:      
                DATA.append(i['_source'])
        except:
            print('CHECK IF THE DATABASE IS ACTIVE')
        return DATA

    def READFILTERUNIQUE(self, emin, emax, host, port, index, size):
        DATA = []
        es = ES.Elasticsearch([{'host': host, 'port': port}])
        querys = {
            "size":0,
            "sort":{
                "title_date": "desc"
            },
            "query": {
                "range": {
                "title_date": {
                    "gte": emin,
                    "lte": emax
                    }
                }
            },
            "aggs": {
                "unique":{
                    "terms": {
                        "field":"title_idcam",
                        "size":size
                    },
                    "aggs": {
                        "unique":{
                            "terms": {
                                "field":"title_identy",
                                "size":size
                                }
                            }
                        }
                    }
                }
            }
        try:
            data  = es.search(index=index, body=querys)
            #  print(data['aggregations']['unique']['buckets'])
            for i in data['aggregations']['unique']['buckets']:
                camara = i['key']
                for j in i['unique']['buckets']:
                    document = j['key']
                    if j['key']!=0:
                        datakey = self.READFILTERDATEDOCUMENTCAM(document=document,emin=emin,emax=emax,cam=camara,host=host,port=port,index=index,size=1)[0] 
                        DATA.append(datakey)
        except:
            print('CHECK IF THE DATABASE IS ACTIVE')

        return DATA
    
    def READFILTEREMOVE(self, document, host, port, index, size):
        DATA = []
        es = ES.Elasticsearch([{'host': host, 'port': port}])
        DATA = self.READFILTERDOCUMENT(document=document, host=host,
            port=port,index=index,size=size
        )
        try:
            for i in DATA:
                es.delete(id=i['_id'], index=index)
            DATA = "REMOVED PERSON " + str(document)
        except:
            DATA = 'CHECK IF THE DATABASE IS ACTIVE'
        return DATA
# DETECCION Y ALINEACIÓN DE ROSTRO    
    def ALIGNMENTANDFACE(self, i, config):
        DATA = []        
        h,w,c = np.shape(i)           
        #La entrada multiplo de 32  
        scale = [512, 512] #scale = [704, 704] para 1080 -> 1052 
        #i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)                   
        Image, f = self.RESIZE_IMAGE(i, scale)            
        Image = np.expand_dims(Image, axis=0)
        Image = Image.astype(np.float32)
        # Model ONNX
        Input_Name = self.artifacts.alignface.get_inputs()[0].name
        RBOX = self.artifacts.alignface.run(None, {Input_Name: Image})
        RBOX = [RBOX[7],RBOX[1],RBOX[4],RBOX[8],RBOX[0],RBOX[3],RBOX[6],RBOX[2],RBOX[5]]

        decay=0.5
        threshold = config['thr'] # thr= 0.75
        nms_threshold = 0.15

        proposals_list = []
        landmarks_list = [] 
        scores_list    = []
                
        _anchors_fpn = {'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
                        'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
                        'stride8':  np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
                        }
        _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}            
        _feat_stride_fpn = [32, 16, 8]
        net_out = [elt for elt in RBOX]            
        sym_idx = 0

        for _idx, s in enumerate(_feat_stride_fpn):
            _key = 'stride%s'%s
            scores = net_out[sym_idx]
            scores = scores[:, :, :, _num_anchors['stride%s'%s]:]

            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

            A = _num_anchors['stride%s'%s]
            K = height * width 
            anchors_fpn = _anchors_fpn['stride%s'%s]
            anchors = self.ANCHORS_PLANE(height, width, s, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            # Scores
            scores  = scores.reshape((-1, 1))
            # Boxes
            bbox_stds = [1.0, 1.0, 1.0, 1.0]
            bbox_deltas = bbox_deltas
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]                

            proposals = self.BBOX_PRED(anchors, bbox_deltas)
            proposals = self.CLIP_BOXES(proposals, np.shape(i))

            if (s == 4 and decay < 1.0):
                scores *= decay

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:,0:4] /= f
            proposals_list.append(proposals)
            scores_list.append(scores)
            # Landmark
            landmark_deltas = net_out[sym_idx + 2]
            landmark_pred_len = landmark_deltas.shape[3]//A
            landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
            landmarks = self.LANDMARK_PRED(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            landmarks[:, :, 0:2] /= f
            landmarks_list.append(landmarks)
            sym_idx += 3

        proposals = np.vstack(proposals_list) # Consolida BOX Vector

        if proposals.shape[0]==0:
            landmarks = np.zeros( (0,5,2) )
            return np.zeros( (0,5) ), landmarks
        scores = np.vstack(scores_list) # Une SCORE Vector
        scores_ravel = scores.ravel() # Todos los valores en un Vector ravel()
        order = scores_ravel.argsort()[::-1]
        # Filta BOX:Proposals, SCORES:scores y LANDMARKS:landmarks
        # basado en la 
        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
        keep = self._NMS(pre_det, nms_threshold)

        det = np.hstack( (pre_det, proposals[:,4:]) )
        det = det[keep, :]
        landmarks = landmarks[keep]

        RESP = {} #variable que contendra los datos json de 
        #rostro
        for idx, face in enumerate(det):
            label = 'face_'+str(idx+1)
            RESP[label] = {}
            RESP[label]["label"] = label
            RESP[label]["score"] = face[4]

            RESP[label]["facial_area"] = list(face[0:4].astype(int))

            RESP[label]["landmarks"] = {}
            RESP[label]["landmarks"]["right_eye"] = list(landmarks[idx][0].astype(int))
            RESP[label]["landmarks"]["left_eye"] = list(landmarks[idx][1].astype(int))
            RESP[label]["landmarks"]["nose"] = list(landmarks[idx][2].astype(int))
            RESP[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3].astype(int))
            RESP[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4].astype(int))

            box = list(face[0:4].astype(int))
            ROI = i[box[1]:box[3], box[0]:box[2]]
           
            RROI = self.ALIGNMENTFACES(
                        w=w, 
                        h=h, 
                        img=ROI,
                        area_facial=list(face[0:4].astype(int)), 
                        right_eye=list(landmarks[idx][0].astype(int)), 
                        left_eye=list(landmarks[idx][1].astype(int)), 
                        nose=list(landmarks[idx][2].astype(int)),
                        mouth_right=list(landmarks[idx][3].astype(int)), 
                        mouth_left=list(landmarks[idx][4].astype(int)),
                        thrcw=config['thrcw'], 
                        thrch=config['thrch']
                        )

            if RROI != []:
                RESP[label]["angles"] = RROI[1]
                RROI = np.array(RROI[0])
                DATA.append([ROI, RROI, RESP[label]])

        return [DATA]
# CODIFICACIÓN   
    def FACE_ENCODING(self, i):
        Image = self.PREPROCESSENCODE(i, dim=160)
        Image = Image/255
        Input_Name = self.artifacts.facesenco.get_inputs()[0].name
        ENCODE = self.artifacts.facesenco.run(None, {Input_Name: Image})
        # ENCODE: Array de 1x128
        return [ENCODE]
# DRAWFACES
    @bentoml.api(input=AnnotatedImageInput(image_input_name='image', annotation_input_name='annotations',
                 accept_image_formats=None, pilmode='RGB'), mb_max_latency=MaxLantencia, 
                                           mb_max_batch_size=MaxBatch, batch=True)
    def DRAWPROPERTIES(self, Image_Array: 'Sequence[imageio.core.util.Array]',
                         Annot_Array: 'Sequence[JsonSerializable]') -> Sequence[str]:
        PROPERTIESARRAY = []
        for img, params in zip(Image_Array, Annot_Array): 
            FACES = self.ALIGNMENTANDFACE(img, params)
            FACES = FACES[0]
            for bface, rface, annot in FACES:
                PROPERTIESARRAY.append(annot)
        return [PROPERTIESARRAY]
# INTEGRACION
    @bentoml.api(input=AnnotatedImageInput(image_input_name='image', annotation_input_name='annotations',
                 accept_image_formats=None, pilmode='RGB'), mb_max_latency=MaxLantencia, 
                                           mb_max_batch_size=MaxBatch, batch=True)
    def FACE_INTEGRATION(self, Image_Array: 'Sequence[imageio.core.util.Array]',
                         Annot_Array: 'Sequence[JsonSerializable]') -> Sequence[str]:
        ENCODING_ARRAY = []
        for img, params in zip(Image_Array, Annot_Array):  
            flagenv = '' 
            # Detecta y alinea rostros salida:Roi,rRoi,params    
            FACES = self.ALIGNMENTANDFACE(img, params)
            FACES = FACES[0]            
            environmentname = "env_"+params['idc']+str(uuid.uuid4())+".jpg"
            for bface, rface, annot in FACES:
                x,y,c = np.shape(rface)
                delta = cv2.Laplacian(bface, cv2.CV_64F).var()
                if x > params['size'] and y > params['size'] and delta > params['desf']:
                    if flagenv == '':
                        ienv = Image.fromarray(img)
                        #ienv.save(f'/home/ossun/sfis/static/environment/{environmentname}')
                        ienv.save(f'/var/opt/sfis/static/environment/{environmentname}')
                        flagenv = environmentname
                # Codifica
                    ENCODING = self.FACE_ENCODING(i=rface)
                    ENCODEFACE = ENCODING[0][0][0]
                # Guarda imagen de rostro
                    face = imutils.resize(bface, height=200)
                    facename = "face_" + params['idc'] + str(uuid.uuid4()) + ".jpg"
                    face = Image.fromarray(face)
                    #  face.save(f'/home/ossun/sfis/static/activity/{facename}')
                    face.save(f'/var/opt/sfis/static/activity/{facename}')
                # Lectura ES
                    try:
                        TASK_READ_ES = self.READES(params, ENCODEFACE)
                        # print(TASK_READ_ES)
                        ENCODING_ARRAY.append(TASK_READ_ES)
                # Escritura ES
                        self.WRITEES(jsonlist=TASK_READ_ES, params=params,
                                    env=environmentname, face=facename, box_face=[0,0,0,0])
                        print('ACTIVITY LOG OK')
                    except:
                        print('CHECK STATUS DATABASE')
                # Acumula RESP
                    ENCODING_ARRAY.append(annot)
            #gc.collect()
        return [ENCODING_ARRAY]
# ESCRITURA de personas en DB ElasticSearch
    @bentoml.api(input=AnnotatedImageInput(image_input_name='image', annotation_input_name='annotations',
                accept_image_formats=None, pilmode='RGB'), mb_max_latency=MaxLantencia, mb_max_batch_size=MaxBatch, batch=True)
    def REGISTRO(self, Image_Array: 'Sequence[imageio.core.util.Array]',
                 Annot_Array: 'Sequence[JsonSerializable]') -> Sequence[str]:
        RESP = []
        for img, params in zip(Image_Array, Annot_Array):  
            # Detecta y alinea rostros     
            FACES = self.ALIGNMENTANDFACE(i=img, config=params)
            FACES = FACES[0]            
            for bface, rface, annot in FACES:
                x,y,c = np.shape(rface)
                if x > params['size'] and y > params['size']:                    
                    ENCODING = self.FACE_ENCODING(rface)
                    ENCODEFACE = ENCODING[0][0][0]
                    # WRITE en DB Elasticsearch
                    es = ES.Elasticsearch(
                        [{'host': params['host'], 'port': params['port']}])
                    index = params['indexwrite']
                    doc = {"title_date": dt.now(), "title_name": params['name'], "title_identy": params['CC'],
                        "title_authorization": params['Access'], "title_vector": ENCODEFACE
                        }
                    try:
                        es.indices.refresh(index=index)
                        count = es.cat.count(index=index, params={
                                            "format": "json"})[0]['count']
                        RESP = es.create(index=index, id=count, body=doc)
                        if RESP["_shards"]['successful'] == 1:
                            print('ALMACENAMIENTO REALIZADO')
                        else:
                            print('IMPOSIBLE ESCRITURA')
                    except:
                        print('CHECK IF THE DATABASE IS ACTIVE')

        return [RESP]
# LECTURA de actividad en DB Elasticsearch
    @bentoml.api(input=JsonInput(), batch=True)
    def READFILTRO(self, jsonlist: List[JsonSerializable]) -> List[str]:
        DATA = []
        for params in jsonlist:
            if   params['search'] in "FECHA":
                DATA = self.READFILTERDATE(emin=params['fmin'], emax=params['fmax'], host=params['host'],
                    port=params['port'],index=params['indexread'],size=params['sizedataread']
                )
            elif params['search'] == "CC":
                DATA = self.READFILTERDOCUMENT(document=params['CC'], host=params['host'],
                    port=params['port'],index=params['indexread'],size=params['sizedataread']
                ) 
            elif params['search'] == "FECHA&CC":
                DATA = self.READFILTERDATEDOCUMENT(document=params['CC'],emin=params['fmin'], 
                    emax=params['fmax'], host=params['host'],port=params['port'],
                    index=params['indexread'],size=params['sizedataread']
                )
            elif params['search'] == "CAMDATEDOC":
                DATA = self.READFILTERDATEDOCUMENTCAM(document=params['CC'],emin=params['fmin'], 
                    emax=params['fmax'],cam=params['cam'],host=params['host'],port=params['port'],
                    index=params['indexread'],size=params['sizedataread']
                )
            elif params['search'] == "CAMDATE":
                DATA = self.READFILTERDATECAM(emin=params['fmin'], emax=params['fmax'], 
                    cam=params['cam'], host=params['host'],port=params['port'],
                    index=params['indexread'], size=params['sizedataread']
                )
            elif params['search'] == "REMOVE":
                DATA = self.READFILTEREMOVE(document=params['CC'], host=params['host'],
                    port=params['port'],index=params['indexread'],size=params['sizedataread']
                )
            elif params['search'] == 'UNIQUE':
                DATA = self.READFILTERUNIQUE(emin=params['fmin'],emax=params['fmax'],
                    host=params['host'],port=params['port'],index=params['indexread'],
                    size=params['sizedataread']
                    )
        return[DATA]
'''# LECTURA db espejo - activity Elasticsearch
    @bentoml.api(input=JsonInput(), batch=True)
    def READAGGS(self, jsonlist: List[JsonSerializable], output=DefaultOutput()) -> List[str]:        
        DATA = [] 
        for params in jsonlist:
            es = ES.Elasticsearch([{'host': params['host'], 'port': params['port']}])
            index = params['indexwrite']

            DATA = self.READFILTERUNIQUE(emin=params['fmin'],emax=params['fmax'],
                host=params['host'],port=params['port'],index=params['indexread'],
                size=params['sizedataread']
            )
            for i in DATA:
                doc = {"title_date": i['title_date'], "title_idcam": i['title_idcam'], "title_name": i['title_name'],
                    "title_identy": i['title_identy'], "title_score": float(i['title_score']), 
                    "title_authorization": i['title_authorization'], "title_imagen_uuid": i['title_imagen_uuid"'], 
                    "title_face_uuid": i['title_face_uuid'], "title_box_face": i['title_box_face']
                }
                try:
                    # refrest id
                    es.indices.refresh(index=index)
                    count = es.cat.count(index=index, params={"format": "json"})[0]['count']
                    # Write aggs
                    es.create(index=index, id=params['idc'] + '_' + count, body=doc)
                except:
                    pass       
        return [DATA]'''
