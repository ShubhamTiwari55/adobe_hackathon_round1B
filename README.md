# Adobe India Hackathon: Persona-Driven Document Intelligence

This repository contains the solution for **Round 1B: Persona-Driven Document Intelligence**. The project is an intelligent system that analyzes a collection of PDF documents and extracts the most relevant sections based on a specific user persona and their "job-to-be-done."

## Overview

The core of this solution is a two-stage machine learning pipeline designed for both speed and accuracy. It first uses a fast retrieval model to identify a broad set of potentially relevant pages from all documents. It then uses a more powerful re-ranking model to score these candidates precisely, identifying not only the most important pages but also the most relevant subsections within those pages.

The entire solution is packaged in a Docker container for easy and reproducible execution in an offline, CPU-only environment, adhering to the challenge constraints.

## Key Features

  * **Two-Stage Ranking Architecture**: Combines a fast LightGBM classifier for initial retrieval with a highly accurate Cross-Encoder for final re-ranking.
  * **Hybrid Feature Set**: The retrieval model uses a blend of semantic, lexical, and structural features for robust candidate selection.
  * **Subsection-Level Granularity**: The system pinpoints and ranks relevant subsections within top pages to provide granular insights, directly addressing the "Sub-Section Relevance" scoring criterion.
  * **OCR Fallback**: Utilizes Tesseract OCR to extract text from image-based or non-standard PDFs, ensuring wider document compatibility.
  * **Dockerized for Reproducibility**: The entire environment is containerized to guarantee consistent execution, as required by the submission guidelines.

## Project Structure

```
.
├── models/                   # Stores downloaded transformer models for offline use
├── sample_test/              # Contains sample input and output for testing
│   ├── input/
│   └── output/
├── approach_explanation.md   # Detailed explanation of the technical approach
├── Dockerfile                # Instructions to build the container
├── download_models.py        # Script to download necessary models
├── lgbm_relevance.joblib     # Pre-trained LightGBM model for candidate retrieval
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── run.py                    # Main execution script
```

## Setup and Execution

### Prerequisites

  * [Docker](https://www.docker.com/get-started) must be installed and running.

### Step 1: Download AI Models

Before building the container, you must download the necessary models for offline use. Run the following command from the project's root directory:

```sh
python download_models.py
```

This will populate the `models` directory.

### Step 2: Build the Docker Image

Build the Docker image using the provided `Dockerfile`. This command packages your code, models, and all dependencies into a single image.

```sh
docker build -t my-solution-1b .
```

### Step 3: Run the Solution

To run the container, you will map a local input directory and a local output directory to the `/app/input` and `/app/output` paths inside the container. The following command processes the data in the `sample_test` folder.

**For Windows (Command Prompt):**

```cmd
docker run --rm -v "%cd%\sample_test\input:/app/input" -v "%cd%\sample_test\output:/app/output" my-solution-1b
```

**For Windows (PowerShell):**

```powershell
docker run --rm -v "${PWD}\sample_test\input:/app/input" -v "${PWD}\sample_test\output:/app/output" my-solution-1b
```

**For Linux or macOS:**

```sh
docker run --rm -v "$(pwd)/sample_test/input:/app/input" -v "$(pwd)/sample_test/output:/app/output" my-solution-1b
```

After the container finishes, the resulting JSON files will be available in the local `sample_test/output` directory.

## License

MIT License

---

For detailed architecture and methodology, see [approach_explanation.md](approach_explanation.md).
