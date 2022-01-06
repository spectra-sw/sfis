from faceservingfixface import FaceOnnx

svc = FaceOnnx()
svc.pack('labels', ['face'])
#svc.pack('facesview', 'version-RFB-640.onnx')
svc.pack('alignface', 'retinafaceseg.onnx')
svc.pack('facesenco', 'facenet512.onnx')
save_path = svc.save()
save_path