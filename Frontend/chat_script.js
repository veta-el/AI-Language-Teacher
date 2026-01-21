const socket = new WebSocket("ws://localhost:3000");
let waitAnswer = false;

// Connect with server
socket.onopen = () => {
    console.log("Connected to server");
};

socket.onerror = (error) => {
    console.error("Error:", error);
};

window.onload = function () { // Load lang_level list selection
    // Add levels to list
    const levels = ["Auto", "A1", "A2", "B1", "B2", "C1", "C2"];

    for (let level = 0; level < 7; level++) {
        let options_level = document.createElement("OPTION");
        document.getElementById("select_lang_level").appendChild(options_level).innerHTML = levels [level];
    };
};

socket.onmessage = (event) => {
    // Handle incoming messages from the server
    console.log(`Received message from server: ${event.data}`);

    try { //Parse text-audio
        const response = JSON.parse(event.data);
        if ("message" in response){ //Make it as text answer
            document.getElementById("message-container").innerHTML += `<div class="msgresp">${response ['message']}</div>`;
        }
        else if ("add_answer" in response){ //Make it as additional answers block
            document.getElementById("user_input").placeholder = "   "+response ['add_answer'];
        }
        else { //Make it as translations
            if (document.getElementById("param_rus").checked) {
                document.getElementById("message-container").innerHTML += `<div class="translation">${response ['rus']}</div>`;
            }
        }
    } catch (SyntaxError) { //Then it is audio
        const audioUrl = URL.createObjectURL(event.data);
        document.getElementById("message-container").innerHTML += `<div class="audioresp"><audio controls><source src="${audioUrl}" type="audio/wav" /></audio></div>`;
    }

    document.getElementById("user_input").removeAttribute("disabled"); //Allow sending new msg
    document.getElementById("wait_animation").style.display = "none";
    waitAnswer = false;

    var scrollDiv = document.getElementById("message-container"); //Scroll msg
    scrollDiv.scrollTo(0, scrollDiv.scrollHeight);
};

socket.onclose = () => {
    console.log("Disconnected from server");
};

// Send data to the backend when the user submits their input

function submitUserInput() {
    //Get all input and additional params
    var user_input = document.getElementById("user_input").value;
    document.getElementById("user_input").value = "";
    var ch_mist_input = document.getElementById("param_check_mistakes").checked;
    var ch_gen_answers = document.getElementById("param_gen_answers").checked;
    var ch_audio_output = document.getElementById("param_audio_output").checked;
    var ch_rus = document.getElementById("param_rus").checked;
    var ch_level = document.getElementById('select_lang_level').options[document.getElementById('select_lang_level').selectedIndex].text;

    // Display input msg
    document.getElementById("message-container").innerHTML += `<div class="msginp">${user_input}</div>`;

    if (user_input !== "") { // Check if there's actual text before sending
        socket.send(JSON.stringify({ audio: "false", user_input: user_input , params: {ch_mist_input: ch_mist_input, ch_gen_answers: ch_gen_answers, ch_audio_output: ch_audio_output, ch_rus: ch_rus, ch_level: ch_level}}));

        document.getElementById("user_input").setAttribute("disabled", "disabled"); //Disable sending new msg before answer from server
        document.getElementById("wait_animation").style.display = "inline";
        waitAnswer = true;
    }
}

document.getElementById("user_input").addEventListener("keypress", (event) => { //Keypress as enter-button
    if (event.key === "Enter") {
        submitUserInput();
    }
});


function toggleImageRec() { //Toggle mic-buttons when rec
    if (!waitAnswer) {
        var element = document.getElementById("user_input_rec");
        var currentClassName = element.className;
  
        if (currentClassName.includes("mic-button-active")) {
            element.className = "mic-button f_near_right_2";
        } else {
            element.className = "mic-button-active f_near_right_2";
        }
    }
}

function startRecording() { //Audio rec
    if (!waitAnswer) { //If we wait for the input from user
        // Get additional params
        var ch_mist_input = document.getElementById("param_check_mistakes").checked;
        var ch_gen_answers = document.getElementById("param_gen_answers").checked;
        var ch_audio_output = document.getElementById("param_audio_output").checked;
        var ch_rus = document.getElementById("param_rus").checked;
        var ch_level = document.getElementById('select_lang_level').options[document.getElementById('select_lang_level').selectedIndex].text;

        if (document.getElementById("user_input_rec").className.includes("mic-button-active")) { //If rec button is active
            navigator.mediaDevices.getUserMedia({ audio: true})
            .then(stream => { //Recording
                const mediaRecorder = new MediaRecorder(stream);
                let voice = [];

                console.log("Recording started");
                mediaRecorder.start();

                mediaRecorder.addEventListener("dataavailable",function(event) { //Push data
                    voice.push(event.data);
                });

                let recordingTimeout = setTimeout(function() { //Timeout 30 sec
                    console.log("Recording time exceeded");
                    mediaRecorder.stop();
                    console.log("Recording stopped");
                    document.getElementById("user_input_rec").className = "mic-button f_near_right_2"
                }, 30000);

                document.getElementById("user_input_rec").addEventListener("click", function(){ //Stop rec if button clicked
                    mediaRecorder.stop();
                    console.log("Recording stopped");
                });

                mediaRecorder.addEventListener("stop", function() { //Send audio data
                    //Display as user input
                    const voiceBlock = new Blob(voice, {type: "audio/wav"});
                    const audioUrl = URL.createObjectURL(voiceBlock);
                    document.getElementById("message-container").innerHTML += `<div class="audioinp"><audio controls><source src="${audioUrl}" type="audio/wav" /></audio></div>`;
                
                    //Send audio to server signal and audio bytes
                    socket.send(JSON.stringify({ audio: "true", user_input: "audio", params: {ch_mist_input: ch_mist_input, ch_gen_answers: ch_gen_answers, ch_audio_output: ch_audio_output, ch_rus: ch_rus, ch_level: ch_level}}))
                    socket.send(voiceBlock)
                
                    document.getElementById("user_input").setAttribute("disabled", "disabled"); //Disable sending new msg before answer from server
                    document.getElementById("wait_animation").style.display = "inline";
                    waitAnswer = true;

                    console.log("Recording sent");
                });
            });
        }
        
    }
}
