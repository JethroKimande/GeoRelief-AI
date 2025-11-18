// Additional JavaScript for GeoRelief-AI frontend
// This file can be extended with additional interactive features

// Example: Add refresh functionality
function refreshData() {
    fetch('/api/get_priority_scores')
        .then(response => response.json())
        .then(data => {
            // Update the map with new data
            console.log('Data refreshed');
        })
        .catch(error => {
            console.error('Error refreshing data:', error);
        });
}

// Export for use in other scripts if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { refreshData };
}

