package main

import (
    "encoding/json"
    "fmt"
    "html/template"
    "io"
    "net/http"
    "os"
    "os/exec"
    "path/filepath"
    "strings"
)

// Define a struct to match the JSON structure of your analysis results
type AnalysisResults struct {
    Accuracy             float64 `json:"accuracy"`
    Precision            float64 `json:"precision"`
    Recall               float64 `json:"recall"`
    F1Score              float64 `json:"f1_score"`
}

// Define a global template variable to hold the upload form and results template
var templates = template.Must(template.ParseFiles("templates/index.html", "templates/results.html"))

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

// handleUpload processes the uploaded file and model name, calling the Python scripts accordingly, then shows the analysis.
func handleUpload(w http.ResponseWriter, r *http.Request) {
    r.ParseMultipartForm(10 << 20) // 10 MB

    file, _, err := r.FormFile("datafile")
    if err != nil {
        http.Error(w, "Invalid file", http.StatusBadRequest)
        return
    }
    defer file.Close()

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

    modelname := r.FormValue("modelname")
    mainPyPath := filepath.Join("..", "python", "main.py")
    analysisPyPath := filepath.Join("..", "python", "analysis.py")

    // Call the main.py script
    cmd := exec.Command("python", mainPyPath, modelname, tempFile.Name())
    cmdOutput, err := cmd.CombinedOutput()
    outputLines := strings.Split(strings.TrimSpace(string(cmdOutput)), "\n")
    fmt.Println(outputLines)
    if err != nil {
        fmt.Printf("Error running main.py: %v\n", err)
        http.Error(w, "Error processing file", http.StatusInternalServerError)
        return
    }
    outputFilePath := outputLines[len(outputLines)-1]

    // Call the analysis.py script
    cmd = exec.Command("python", analysisPyPath, outputFilePath)
    cmdOutput, err = cmd.CombinedOutput()
    fmt.Printf("Python script output: %s\n", string(cmdOutput))
    if err != nil {
        fmt.Printf("Error running analysis.py: %v\n", err)
        http.Error(w, "Error processing file with analysis.py", http.StatusInternalServerError)
        return
    }

    // Parse the analysis results
    outputFilePath = strings.TrimSuffix(outputFilePath, ".json") + "_results.json"
    fmt.Println(outputFilePath)
    analysisResults, err := parseAnalysisResults(outputFilePath)
    if err != nil {
        fmt.Printf("Error parsing analysis results: %v\n", err)
        http.Error(w, "Error processing analysis results", http.StatusInternalServerError)
        return
    }

    // Render the results template with the analysis data
    if err := templates.ExecuteTemplate(w, "results.html", analysisResults); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
    }
}

// parseAnalysisResults parses the JSON results file into an AnalysisResults struct.
func parseAnalysisResults(filePath string) (AnalysisResults, error) {
    var results AnalysisResults
    file, err := os.ReadFile(filePath)
    if err != nil {
        return results, err
    }
    err = json.Unmarshal(file, &results)
    return results, err
}





