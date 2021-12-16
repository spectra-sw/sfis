function menu(){
    window.open('/menu','_self');
}
function inicio(){
    window.open('/inicio','_self');
}
function iniciar(){
  
        url = '/activity/get'
        $.ajax({
                url: url,
                type:'GET',
                success: function(data) {
                    $("#registros").html(data);
                }
        });
        url = '/faces/get'
        $.ajax({
                url: url,
                type:'GET',
                success: function(data) {
                    $("#faces").html(data);
                }
        });
}
//CONFIG00:Funcion que actualiza el estado de camara
function stcamara(){
    $('#tablalistacam tbody tr').each(function(){
        idcam = $(this).find('td:eq(11)').text();
        orden = $(this).find('td:eq(0)').text();
        url = '/estadodecamara/estado/'+idcam+'&'+orden;
        //console.log(url);
        $.ajax({
            url:url,
            type:'GET',
            success:function(data){
                etiqueta ='#' + $(data).attr('id')
                $(etiqueta).html(data)
                //console.log(data, etiqueta)
            },
        });
    });
}
//CONFIG01:Gestión de streaming para configuracion
function pruebas(id){
    console.log(id);
}

var streamingmodalcamara;
function startintervalstreaming(recipiente){
        streamingmodalcamara = setInterval(function(){
            viewstreamingcamara(recipiente)
        }, 200)
}
function stopviewstreamingcamara(id){
    url_serve_camara = '/activar/'+id+'/0';
    $.ajax({
        url: url_serve_camara,
        type: 'GET',
        success:function(calibration) {
           console.log(calibration); 
        }
    });
}
function viewstreamingcamara(id){
    url_video_feed = '/setid/'+id
    url_serve_camara = '/activar/'+id+'/1';
    $.ajax({
        url: url_video_feed,
        type: 'GET',
        success: function(id){
            console.log(id);
        } 
    });
    $.ajax({
        url: url_serve_camara,
        type: 'GET',
        success:function(calibration) {
            console.log(calibration);
            
        }
    });
}
function viewtest(recipiente){
    url = 'http://127.0.0.1:8000/WEB_'+recipiente+'/video_feed'
    console.log(url)
    $.ajax({
        url: url,
        type: 'GET',
        success:function(data){

        }
    });
}
 
function detalleA(id){
    url = '/activity/get/'+id
        $.ajax({
                url: url,
                type:'GET',
                success: function(data) {
                    $("#detalleA").html(data);
                    $("#detalleA").modal();
                }
        }); 
   
}
function detalleA2(id){
    url = '/activity/get2/'+id
        $.ajax({
                url: url,
                type:'GET',
                success: function(data) {
                    $("#detalleA2").html(data);
                    $("#detalleA2").modal();
                }
        }); 
   
}
function buscarp(){
    tipoid = $("#tipoid").val();
    id = $("#id").val();
    url = '/buscarp'
    
    datos ={ tipoid: tipoid,id:id}
        $.ajax({
                url: url,
                data: datos,
                type:'POST',
                success: function(data) {
                    $("#datosp").html(data);
                },
                error: function(error) {
                    console.log(error);
                }
        }); 
}

function camaractualizar(id, accion){
    var arrayname = id.split("&")
    var id = arrayname[0];
    var uuid = arrayname[1];
    url = '/updatecamara'
    url_serve_camara = '/activar/'+uuid+'/1';
    console.log(id);
    parametros = $( "#actualizacionparametros" ).serialize(); 
    datos ={ id:id, parametros:parametros, accion: accion}
    $.ajax({
            url: url,
            data: datos,
            type:'POST',
            success: function(data) {
                $.ajax({
                    url: url_serve_camara,
                    type: 'GET',
                    success:function(calibration) {
                        console.log(calibration); 
                    }
                });
            },
            error: function(error) {
                console.log(error);
            }
    }); 
}

function camara(id,accion){
    url = '/activarcamara'
    parametros = $( "#parametros" ).serialize(); 
    
    datos ={ id:id, parametros:parametros, accion: accion}
    //console.log(datos)
        $.ajax({
                url: url,
                data: datos,
                type:'POST',
                success: function(data) {
                    alert(data)
                    /*
                    if (data == id){
                        alert("CÁMARA ACTIVADA");
                    }
                    else{
                        alert("ERROR AL ACTIVAR CÁMARA");
                    }*/
                },
                error: function(error) {
                    console.log(error);
                }
        }); 
}

function testcamara(id){
    url = '/testcamara'
    
    datos ={ id:id}
    //console.log(datos)
        $.ajax({
                url: url,
                data: datos,
                type:'POST',
                success: function(data) {
                    alert(data)
                },
                error: function(error) {
                    console.log(error);
                }
        }); 
}
function guardarCamara(){
    data=$( "#formCamara" ).serialize(); 
    
    url="/guardarcamara";
    $.ajax({
        url: url,
        data:data,
        type: "POST", 
        success: function(data){
            alert(data)
          }
      });
}
function server(status){
    data ={status : status}
    url="/server";
    $.ajax({
        url: url,
        data: data,
        type: "POST", 
        success: function(data){
            alert(data)
          }
      });
}
function buscaract(){
    data=$( "#formbuscaract" ).serialize();  
    url="/buscaract";
    $.ajax({
        url: url,
        data:data,
        type: "POST", 
        success: function(data){
            $("#resultado").html(data);
          }
      });
}
