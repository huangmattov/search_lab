function ShowCam() {
    Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 100
    });
    Webcam.attach('#my_camera');
}
// fetch the login_form.html template and embed it into content div
//function loadForm() {
//    var xhr= new XMLHttpRequest();
//    xhr.open('GET', '/login_form/');
//    xhr.onreadystatechange= function() {
//        if (this.readyState!==4) return;
//        if (this.status!==200) return;
//        document.getElementById('content').innerHTML = this.responseText;
//    };
//    xhr.send();
//}
// bind loadForm() and ShowCam() functions to the corresponding events
window.addEventListener("DOMContentLoaded", loadForm);
window.addEventListener("load", ShowCam);
function snap() {
    Webcam.snap( function(data_uri) {
        // display results in page
        document.getElementById('results').innerHTML =
        '<img id="image" src="'+ data_uri+'"/>';
      } );
}
function upload() {
    var photo = document.getElementById('image').src;
    //var form = document.getElementById('loginForm');
    //photo = dataURItoBlob(photo)
    //var formData = new FormData(form);
    //formData.append("file", photo);
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function()
        {
          if(this.readyState == 4 && this.status == 200) {
            document.getElementById('content').innerHTML = this.responseText;
            ShowCam();
          } else {
            document.getElementById('content').innerHTML = "ERROR";
          }
        }
    xmlhttp.open("POST", "/get_nearest_neighbour/", false);
    xmlhttp.send(photo);
}
function dataURItoBlob(dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = unescape(dataURI.split(',')[1]);
    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ia], {type:mimeString});
}