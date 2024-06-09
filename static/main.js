window.onload = function() {
    const recognition = new webkitSpeechRecognition();
    let mediaRecorder;
    let recordedAudioChunks = [];

    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = function(event) {
        const speechToText = event.results[0][0].transcript;
        console.log('Speech recognized:', speechToText);
    };

    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const boxContainer = document.querySelector('.boxContainer');

    startBtn.addEventListener('click', function() {
        recordedAudioChunks = []; // Clear previous recording
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = function(event) {
                recordedAudioChunks.push(event.data);
            };
            mediaRecorder.onstop = function() {
                const audioBlob = new Blob(recordedAudioChunks, { type: 'audio/wav' });
                console.log('Generated .wav file size:', audioBlob.size);
                if (audioBlob.size === 0) {
                    console.error('Empty .wav file generated');
                    return;
                }
                downloadBlobAsFile(audioBlob, 'recorded_audio.wav');
            };
            mediaRecorder.start();
            recognition.start();
            boxContainer.classList.add('recording'); // Add class for animation
            console.log('Speech recognition and recording started');
        }).catch(error => {
            console.error('Error accessing audio stream:', error);
        });
    });

    stopBtn.addEventListener('click', function() {
        recognition.stop();
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            boxContainer.classList.remove('recording'); // Remove class to stop animation
        }
        console.log('Speech recognition and recording stopped');
    });

    function downloadBlobAsFile(blob, fileName) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        setTimeout(() => {
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }, 0);
    }
};