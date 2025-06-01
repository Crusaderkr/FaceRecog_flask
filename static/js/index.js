const videoFeed = document.getElementById('video_feed');
const toast = document.getElementById('toast');

// Start recognition stream
document.getElementById('startBtn').addEventListener('click', () => {
    fetch('/start_stream', { method: 'POST' })
        .then(() => {
            videoFeed.src = "/video_feed";
        })
        .catch(err => {
            console.error("Failed to start stream:", err);
        });
});

// Stop recognition stream
document.getElementById('stopBtn').addEventListener('click', () => {
    fetch('/stop_stream', { method: 'POST' })
        .then(() => {
            videoFeed.src = "./static/images/placeholder.png";
        })
        .catch(err => {
            console.error("Failed to stop stream:", err);
        });
});

// Show attendance toast
function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 3000);
}

// Poll attendance every 2 seconds
setInterval(() => {
    fetch("/latest_attendance")
        .then(res => res.json())
        .then(data => {
            if (data.marked) {
                showToast(`✅ Attendance marked for ${data.name}`);
            }
        })
        .catch(err => console.error("Error checking attendance:", err));
}, 2000);
