package main

import (
	"fmt"
	"html/template"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Define a global template variable to hold the upload form template
var templates = template.Must(template.ParseFiles("templates/index.html"))

func main() {
    // Ensure the temp directory exists
    tempDirPath := "temp" // Relative path to the temp directory
    if err := os.MkdirAll(tempDirPath, os.ModePerm); err != nil {
        fmt.Printf("Error creating temp directory: %v\n", err)
        return // Exit if we can't ensure the temp directory exists
    }

    http.HandleFunc("/", uploadPageHandler)  // Serve the upload form
    http.HandleFunc("/upload", handleUpload) // Process uploads
    fmt.Println("Server is running on http://localhost:8080")
    http.ListenAndServe(":8080", nil)
}

// uploadPageHandler renders the HTML form for uploading files.
func uploadPageHandler(w http.ResponseWriter, r *http.Request) {
    if err := templates.ExecuteTemplate(w, "index.html", nil); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
    }
}

// handleUpload processes the uploaded file and model name, calling the Python scripts accordingly.
func handleUpload(w http.ResponseWriter, r *http.Request) {
    // Limit the size of the incoming request to prevent abuse.
    r.ParseMultipartForm(10 << 20) // 10 MB

    // Parse the uploaded file.
    file, _, err := r.FormFile("datafile")
    if err != nil {
        http.Error(w, "Invalid file", http.StatusBadRequest)
        return
    }
    defer file.Close()

    // Read the file into a temporary file within the temp directory.
    tempFile, err := os.CreateTemp("temp", "upload-*.json")
    if err != nil {
        fmt.Println("Error creating temp file:", err)
        http.Error(w, "Server error", http.StatusInternalServerError)
        return
    }
    defer tempFile.Close()

    _, err = io.Copy(tempFile, file)
    if err != nil {
        fmt.Println("Error copying file:", err)
        http.Error(w, "Server error", http.StatusInternalServerError)
        return
    }

    // Extract the model name from the form.
    modelname := r.FormValue("modelname")

    // Prepare the path for the Python scripts within the python directory.
    mainPyPath := filepath.Join("..", "python", "main.py")
	analysisPyPath := filepath.Join("..", "python", "analysis.py")

    // First, call the main.py script using "python3".
    cmd := exec.Command("python3", mainPyPath, modelname, tempFile.Name())
	cmdOutput, err := cmd.CombinedOutput() // Captures both standard output and standard error
	outputLines := strings.Split(strings.TrimSpace(string(cmdOutput)), "\n")
	// Determine the output file path from main.py.
	outputFilePath := outputLines[len(outputLines)-1]

	// Immediately after executing the command, print the output to Go's console
	fmt.Printf("Python script output: %s\n", string(cmdOutput))

	if err != nil {
		// If there's an error, include that in your log as well
		fmt.Printf("Error running Python script: %v\n", err)
		http.Error(w, "Error processing file", http.StatusInternalServerError)
		return
	}
    

    // Then, call the analysis.py script with the output from main.py using "python3".
    cmd = exec.Command("python3", analysisPyPath, outputFilePath)
    cmdOutput, err = cmd.CombinedOutput()
	fmt.Printf("Python script output: %s\n", string(cmdOutput))
    if err != nil {
        fmt.Printf("Error running analysis.py: %v, output: %s\n", err, string(cmdOutput))
        http.Error(w, "Error processing file with analysis.py", http.StatusInternalServerError)
        return
    }
	fmt.Printf("Python script output: %s\n", string(cmdOutput))

    // After processing, you might want to redirect the user to a "success" page or directly serve the results.
    // For simplicity, here we'll just send a basic response.
    w.Write([]byte("File uploaded and processed successfully."))
}




