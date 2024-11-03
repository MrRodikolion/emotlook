function update_stream() {
    update_frame();
    update_netframe();
    update_diag();
}

const b64toBlob = (b64Data, contentType = '', sliceSize = 512) => {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];

    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
        const slice = byteCharacters.slice(offset, offset + sliceSize);

        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }

        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
    }

    const blob = new Blob(byteArrays, { type: contentType });
    return blob;
}

setInterval(update_stream, 100);

function update_frame() {
    // const norm_url = "{{ url_for('get_frame') }}"
    fetch(norm_url)
        .then(response => response.json())
        .then(data => {
            const imgstr = data.image;

            const blob = b64toBlob(imgstr, 'image/jpg');
            const blobUrl = URL.createObjectURL(blob);

            document.getElementById("normframe").src = blobUrl;
        })
        .catch(error => console.error(error));
}

function update_netframe() {
    // const net_url = "{{ url_for('get_netframe') }}"
    fetch(net_url)
        .then(response => response.json())
        .then(data => {
            const imgstr = data.image;

            const blob = b64toBlob(imgstr, 'image/jpg');
            const blobUrl = URL.createObjectURL(blob);

            document.getElementById("netframe").src = blobUrl;
        })
        .catch(error => console.error(error));
}

var diag = null;
const ctx = document.getElementById('diag').getContext('2d');
function update_diag() {
    // const diag_url = "{{ url_for('get_emotes') }}"
    fetch(diag_url)
        .then(response => response.json())
        .then(data => {
            const names = data.names;
            const values = data.data;

            if (diag !== null) {
                diag.data.labels = names;
                diag.data.datasets[0].data = values;
                diag.update();
            }
            else {
                diag = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: names,
                        datasets: [{
                            label: 'Значения',
                            data: values,
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.8)',
                                'rgba(54, 162, 235, 0.8)',
                                'rgba(255, 206, 86, 0.8)',
                                'rgba(75, 192, 192, 0.8)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        animation: false,
                        scales: {
                            yAxes: [{
                                ticks: {
                                    beginAtZero: true
                                }
                            }]
                        }
                    }
                })
            }
        })
        .catch(error => console.error('Ошибка:', error));

}