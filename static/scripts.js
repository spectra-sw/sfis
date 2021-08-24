function menu(){
    window.open('/menu','_self');
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
function saludo(){
    alert("ghgfhvnbfvhgnbgbhgyuhjiy7ui")
}