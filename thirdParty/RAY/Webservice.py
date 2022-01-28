from flask import Flask, request
from subprocess import check_output

app = Flask(__name__)

@app.route('/')
def init():
    return "<h3 style='color:green'>Active...</h3>"

@app.route('/camservice/active', methods=['GET', 'POST'])
def activecam():
    #data request service
    data = request.json
    #Data JSON
    idc = data['idc']
    source = data['source']
    frame = data['frame']
    urlservices = data['urlservices']
    timeml = int(data['timeml'])
    indexread = data['indexread']
    indexwrite = data['indexwrite']
    namespace = data['namespace']
    nreplica = data['nreplica']
    address = data['address']
    hostname = data['hostname']
    port = data['port']
    sizeread = data['sizeread']
    sizeface = int(data['sizeface'])
    thr = data['thr']
    thrperson = data['thrperson']
    thrminperson = data['thrminperson']
    thrcw = data['thrcw']
    thrch = data['thrch']
    #Line Command
    commandadd = f'python3 /home/service/Serveargument.py --id {idc} --source {source} \
    --frame {frame} --urlservices {urlservices} --timeml {timeml} --indexread {indexread} \
    --indexwrite {indexwrite} --namespace {namespace} --nreplica {nreplica} --address {address} \
    --hostname {hostname} --port {port} --sizeread {sizeread} --sizeface {sizeface} --thr {thr} \
    --thrperson {thrperson} --thrminperson {thrminperson} --thrcw {thrcw} --thrch {thrch} --add True'
    #  print(commandadd)
    answer = check_output(commandadd, shell=True).decode('utf-8')
    #  print(answer)
    return commandadd

@app.route('/camservice/deactive', methods=['GET','POST'])
def deactivecam():
    #data request service
    data = request.json
    #Data JSON
    idc = data['idc']
    source = data['source']
    frame = int(data['frame'])
    urlservices = data['urlservices']
    timeml = data['timeml']
    indexread = data['indexread']
    indexwrite = data['indexwrite']
    namespace = data['namespace']
    nreplica = data['nreplica']
    address = data['address']
    hostname = data['hostname']
    port = data['port']
    sizeread = data['sizeread']
    sizeface = int(data['sizeface'])
    thr = data['thr']
    thrperson = data['thrperson']
    thrminperson = data['thrminperson']
    thrcw = data['thrcw']
    thrch = data['thrch']
    #Line Command
    commandadd = f'python3 /home/service/Serveargument.py --id {idc} --source {source} \
    --frame {frame} --urlservices {urlservices} --timeml {timeml} --indexread {indexread} \
    --indexwrite {indexwrite} --namespace {namespace} --nreplica {nreplica} --address {address} \
    --hostname {hostname} --port {port} --sizeread {sizeread} --sizeface {sizeface} --thr {thr} \
    --thrperson {thrperson} --thrminperson {thrminperson} --thrcw {thrcw} --thrch {thrch} --remove True'
    #  print(commandadd)
    answer = check_output(commandadd, shell=True).decode('utf-8')
    #  print(answer)
    return commandadd

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=7543)
