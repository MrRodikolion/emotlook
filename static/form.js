
function send(e, form) {
  
  // var file = document.getElementById('file').value;

  // var formData = new FormData();
  // formData.append('file1', file);

  // const data = new URLSearchParams(formData);

  fetch(img2netimg_url, {
    method: 'POST',
    body: new FormData(form),
  })
  .then(response => response.json())
  .then(data => {
    const imgstr = data.image;

    const blob = b64toBlob(imgstr, 'image/jpg');
    const blobUrl = URL.createObjectURL(blob);

    document.getElementById("imsin").src = blobUrl;
  })
  .catch(error => console.error('Ошибка:', error));
  
  e.preventDefault();
}
