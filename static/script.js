function showLoadingModal() {
    document.getElementById('loadingModal').style.display = 'flex';
}

window.addEventListener("load", function () {
    const loadingModal = document.getElementById('loadingModal');
    loadingModal.style.display = "none";
});

function closeLightbox(event) {
    event.preventDefault();
    var lightbox = document.getElementById('lightbox');
    lightbox.style.display = 'none';
}
