
document.getElementById('myForm').addEventListener('submit', function(e) {
  e.preventDefault();
  
  var file = document.getElementById('file').value;

  var formData = new FormData();
  formData.append('file1', file);

  fetch('/', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    const imgstr = data.image;

    const blob = b64toBlob(imgstr, 'image/jpg');
    const blobUrl = URL.createObjectURL(blob);

    document.getElementById("imsin").src = blobUrl;
  })
  .catch(error => console.error('Ошибка:', error));
});
