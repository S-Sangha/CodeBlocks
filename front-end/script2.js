function showPopup(color) {
    // Hide all pop-ups
    hideAllPopups();

    // Show the corresponding pop-up
    document.getElementById(`${color}Popup`).style.display = 'block';
}

function hideAllPopups() {
    // Hide all pop-ups
    document.getElementById('redPopup').style.display = 'none';
    document.getElementById('bluePopup').style.display = 'none';
    document.getElementById('greenPopup').style.display = 'none';
    document.getElementById('purplePopup').style.display = 'none';
}
