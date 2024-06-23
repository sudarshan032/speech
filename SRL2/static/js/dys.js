function handleFileUpload(input) {
                    
    var audioPlayer = document.getElementById('audioPlayer1');
                                    
    if (input.files && input.files[0]) 
    {
        var reader = new FileReader();
                        
        reader.onload = function (e) 
        {
            audioPlayer.src = e.target.result;
            audioPlayer.play();
        };
                        
        reader.readAsDataURL(input.files[0]);
    }
}

function startRecording() 
{
    fetch('/start_recording', {
                method: 'POST'
            }).then(response => response.json())
              .then(data => console.log(data));
}

        function stopRecording() {
            fetch('/stop_recording', {
                method: 'POST'
            }).then(response => response.json())
              .then(data => console.log(data));
        }