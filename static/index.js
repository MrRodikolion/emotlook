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

function update_diag() {
    // const diag_url = "{{ url_for('get_emotes') }}"
    fetch(diag_url)
        .then(response => response.json())
        .then(data => {
            const imgstr = data.image;

            const blob = b64toBlob(imgstr, 'image/jpg');
            const blobUrl = URL.createObjectURL(blob);

            document.getElementById("diag").src = blobUrl;
        })
        .catch(error => console.error(error));
}