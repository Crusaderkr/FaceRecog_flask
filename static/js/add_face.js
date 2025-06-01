const uploadForm = document.getElementById('upload_form'); // Corrected ID
const uploadMessage = document.getElementById('upload_message'); // Corrected ID

if (uploadForm) { // Ensure the form exists on the page
    uploadForm.addEventListener('submit', function (event) {
        event.preventDefault();

        const form = this;
        const formData = new FormData(form);
        const messageDiv = uploadMessage; // Use the globally defined variable
        const loadingDiv = document.getElementById('upload_loading');

        // Reset messages
        messageDiv.textContent = '';
        messageDiv.className = '';
        loadingDiv.style.display = 'block';  // Show loading

        fetch(form.action, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            loadingDiv.style.display = 'none';  // Hide loading

            if (data.success) {
                messageDiv.className = 'success';
                messageDiv.textContent = data.success;
                form.reset();
            } else {
                messageDiv.className = 'error';
                messageDiv.textContent = data.error || 'Unknown error.';
            }
        })
        .catch(error => {
            console.error('Upload failed:', error);
            loadingDiv.style.display = 'none';  // Hide loading
            messageDiv.className = 'error';
            messageDiv.textContent = 'Failed to upload face.';
        });
    });
}

// Camera capture setup
const cameraVideo = document.getElementById('camera_preview'); // Corrected ID
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture_button');
const capturedPreview = document.getElementById('preview'); // Corrected ID
const saveBtn = document.getElementById('save_captured_face'); // Corrected ID
const capturedNameInput = document.getElementById('camera_name'); // Corrected ID
const captureMessage = document.getElementById('camera_message'); // Corrected ID
const openCameraButton = document.getElementById('open_camera_button'); // New button ID

let cameraStream = null;

if (openCameraButton) {
    openCameraButton.addEventListener('click', () => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                cameraStream = stream;
                cameraVideo.srcObject = stream;
                cameraVideo.style.display = 'block';
                captureBtn.style.display = 'inline-block';
                openCameraButton.style.display = 'none';
            })
            .catch(err => {
                console.error("Failed to access webcam", err);
                captureMessage.className = 'error';
                captureMessage.textContent = 'Failed to open camera.';
            });
    });
}

if (captureBtn) {
    captureBtn.addEventListener('click', () => {
        const ctx = canvas.getContext('2d');
        canvas.width = cameraVideo.videoWidth;
        canvas.height = cameraVideo.videoHeight;
        ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');
        capturedPreview.src = dataURL;
        capturedPreview.style.display = 'block';
    });
}

if (saveBtn) {
    saveBtn.addEventListener('click', () => {
        const name = capturedNameInput.value.trim();
        if (!name) {
            captureMessage.className = 'error';
            captureMessage.textContent = 'Enter a name.';
            return;
        }

        const dataURL = canvas.toDataURL('image/jpeg');
        const loadingDiv = document.getElementById('camera_spinner'); // Get the spinner

        captureMessage.textContent = ''; // Clear previous messages
        captureMessage.className = '';
        loadingDiv.style.display = 'inline-block'; // Show spinner

        fetch('/add_captured_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, image: dataURL })
        })
        .then(res => res.json())
        .then(data => {
            loadingDiv.style.display = 'none'; // Hide spinner
            captureMessage.className = data.success ? 'success' : 'error';
            captureMessage.textContent = data.success || data.error;
            if (data.success) {
                capturedPreview.style.display = 'none';
                capturedNameInput.value = '';
                if (cameraStream) {
                    cameraStream.getTracks().forEach(track => track.stop());
                    cameraStream = null;
                    cameraVideo.srcObject = null;
                    cameraVideo.style.display = 'none';
                    captureBtn.style.display = 'none';
                    openCameraButton.style.display = 'inline-block';
                }
            }
        })
        .catch(err => {
            console.error(err);
            loadingDiv.style.display = 'none'; // Hide spinner
            captureMessage.className = 'error';
            captureMessage.textContent = 'Failed to save captured face.';
        });
    });
}