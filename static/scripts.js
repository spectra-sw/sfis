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