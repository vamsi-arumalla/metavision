// assets/chunked_upload.js
function setupChunkedUpload() {
    console.log('Setting up chunked upload...');
    
    // First, check if all required elements exist
    var uploadArea = document.getElementById('large-file-upload-area');
    var statusDiv = document.getElementById('chunked-upload-status');
    var progressDiv = document.getElementById('chunked-upload-progress');
    var fileDisplayDiv = document.getElementById('large-file-display');
    var progressBar = document.getElementById('large-file-progress-bar');
    
    console.log('=== Element Check ===');
    console.log('Upload area:', !!uploadArea);
    console.log('Status div:', !!statusDiv);
    console.log('Progress div:', !!progressDiv);
    console.log('File display div:', !!fileDisplayDiv);
    console.log('Progress bar:', !!progressBar);
    console.log('====================');
    
    function initResumable() {
        if (!window.Resumable) {
            console.log('Resumable.js not yet loaded, retrying...');
            setTimeout(initResumable, 100);
            return;
        }
        
        console.log('Resumable.js loaded, initializing...');
    var r = new Resumable({
        target: '/chunk-upload',
        chunkSize: 2 * 1024 * 1024,
        simultaneousUploads: 3,
            testChunks: false,
            fileType: ['csv'],
            maxFiles: 1
    });

    function tryAssignBrowse() {
            var browseArea = document.getElementById('large-file-upload-area');
        var browseButton = document.getElementById('chunked-upload-btn');
            var statusDiv = document.getElementById('chunked-upload-status');
            var progressDiv = document.getElementById('chunked-upload-progress');

            console.log('Looking for upload elements...');
            console.log('Browse area:', browseArea);
            console.log('Browse button:', browseButton);

            if (browseArea) {
                console.log('Assigning browse to large file upload area');
                r.assignBrowse(browseArea);
                clearInterval(interval);

                // Make the area look clickable
                browseArea.style.cursor = 'pointer';

                // Add click handler for better UX
                browseArea.addEventListener('click', function () {
                    console.log('Large file upload area clicked');
                });
            } else if (browseButton) {
                console.log('Assigning browse to chunked upload button');
            r.assignBrowse(browseButton);
            clearInterval(interval);
        }
    }

    var interval = setInterval(tryAssignBrowse, 500);

    r.on('fileAdded', function (file) {
            console.log('File added:', file.fileName, 'Size:', file.size);

            var statusDiv = document.getElementById('chunked-upload-status');
            var progressDiv = document.getElementById('chunked-upload-progress');
            var fileDisplayDiv = document.getElementById('large-file-display');
            var progressBar = document.getElementById('large-file-progress-bar');

            console.log('Status div found:', !!statusDiv);
            console.log('Progress div found:', !!progressDiv);
            console.log('File display div found:', !!fileDisplayDiv);
            console.log('Progress bar found:', !!progressBar);

            if (statusDiv) {
                statusDiv.innerHTML = '<i class="fas fa-upload" style="margin-right: 8px; color: #00bcd4;"></i>Uploading ' + file.fileName + '...';
                console.log('Updated status div');
            }

            if (fileDisplayDiv) {
                fileDisplayDiv.innerHTML = '<i class="fas fa-file-csv" style="margin-right: 8px; color: #4caf50;"></i>' + file.fileName + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
                fileDisplayDiv.style.display = 'block';
                console.log('Updated file display div');
            }

            if (progressBar) {
                progressBar.style.display = 'block';
                console.log('Showed progress bar');
            }

        r.upload();
    });

    r.on('fileSuccess', function (file, message) {
            console.log('File upload success:', file.fileName);

            var statusDiv = document.getElementById('chunked-upload-status');
            var progressText = document.getElementById('large-file-progress-text');
            var progressInner = document.getElementById('large-file-progress-inner');

            if (statusDiv) {
                statusDiv.innerHTML = '<i class="fas fa-check-circle" style="margin-right: 8px; color: #4caf50;"></i>Upload complete! Use "Run Processing Pipeline" to process.';
            }

            if (progressText) {
                progressText.innerText = '100%';
            }

            if (progressInner) {
                progressInner.style.width = '100%';
            }
        });

    r.on('fileError', function (file, message) {
            console.log('File upload error:', message);

            var statusDiv = document.getElementById('chunked-upload-status');
            if (statusDiv) {
                statusDiv.innerHTML = '<i class="fas fa-exclamation-triangle" style="margin-right: 8px; color: #f44336;"></i>Upload failed: ' + message;
            }
    });

    r.on('progress', function () {
        var percent = Math.floor(r.progress() * 100);
            console.log('Upload progress:', percent + '%');

            var progressDiv = document.getElementById('chunked-upload-progress');
            var progressText = document.getElementById('large-file-progress-text');
            var progressInner = document.getElementById('large-file-progress-inner');

            console.log('Progress elements found - Div:', !!progressDiv, 'Text:', !!progressText, 'Inner:', !!progressInner);

            if (progressDiv) {
                progressDiv.innerHTML = '<i class="fas fa-spinner fa-spin" style="margin-right: 8px;"></i>Progress: ' + percent + '%';
                console.log('Updated progress div to:', percent + '%');
            }

            if (progressText) {
                progressText.innerText = percent + '%';
                console.log('Updated progress text to:', percent + '%');
}

            if (progressInner) {
                progressInner.style.width = percent + '%';
                console.log('Updated progress bar width to:', percent + '%');
            }
        });

        console.log('Chunked upload setup complete');
    }

    initResumable();
}

// Make setupChunkedUpload globally available
window.setupChunkedUpload = setupChunkedUpload;

// Initialize when DOM is ready and when Dash components are updated
document.addEventListener('DOMContentLoaded', setupChunkedUpload);

// Also try to initialize after a delay to handle Dash component updates
setTimeout(setupChunkedUpload, 1000);
setTimeout(setupChunkedUpload, 3000);

// Watch for the large file upload area to become visible
var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
            var target = mutation.target;
            if (target.id === 'large-file-upload-area' && target.style.display !== 'none') {
                console.log('Large file upload area became visible, reinitializing...');
                setTimeout(setupChunkedUpload, 200);
            }
        }
    });
});

// Start observing when DOM is ready
document.addEventListener('DOMContentLoaded', function () {
    var uploadArea = document.getElementById('large-file-upload-area');
    if (uploadArea) {
        observer.observe(uploadArea, { attributes: true, attributeFilter: ['style'] });
    } else {
        // If not found immediately, try again later
        setTimeout(function () {
            var uploadArea = document.getElementById('large-file-upload-area');
            if (uploadArea) {
                observer.observe(uploadArea, { attributes: true, attributeFilter: ['style'] });
            }
        }, 1000);
    }
}); 