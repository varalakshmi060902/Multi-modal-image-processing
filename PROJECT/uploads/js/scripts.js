// scripts.js

// Example: Add a confirmation alert before submitting the form
document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    form.addEventListener("submit", function(event) {
        const files = document.querySelector("#files").files;
        if (files.length === 0) {
            alert("Please upload at least one MRI file before submitting.");
            event.preventDefault();  // Prevent form submission
        } else {
            alert("Processing your MRI files...");
        }
    });
});
