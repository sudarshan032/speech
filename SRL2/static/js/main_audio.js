function handleFileUpload(input) {
    var audioPlayer = document.getElementById('audioPlayer');
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            audioPlayer.src = e.target.result;
            audioPlayer.play();
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function startRecording() {
    fetch('/start_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'recording') {
                console.log('Recording started...');
                document.getElementById('record-status').innerText = 'Recording...';
            }
        });
}

function stopRecording() {
    fetch('/stop_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'finished') {
                console.log('Recording stopped.');
                document.getElementById('record-status').innerText = 'Recording stopped.';
                var audioPlayer2 = document.getElementById('audioPlayer2');
                audioPlayer2.src = 'audio_files/output.wav';
                audioPlayer2.play();
                document.getElementById('hidden-file-path').value = 'audio_files/output.wav';
            }
        });
}