# Lecture Files Overview

## What This Lecture is About

- This lecture teaches Natural Language Processing (NLP) from basics to advanced LLMs
- **Goal 1: NLP Basics** - Learn fundamental techniques for processing and analyzing text (including "AI" models)
- **Goal 2: Build an LLM from Scratch** - Understand how modern large language models (LLMs) actually work by building one
- **Goal 3: Production Code** - Study best practices to developdeploy NLP systems in real-world applications

---


**Please review the following notebooks and try running the code BEFORE the lecture:**

## A. Helpful tools to run the code below

-  Make sure you have Python and Jupyter installed on your computer
    - Python version and library versions can be viewed in`pyproject.toml`
        - Use [uv python and library manager](https://docs.astral.sh/uv/getting-started/installation/) (I will show how to use this in class)
- Open the notebook files (.ipynb) in Jupyter and try running them cell by cell
    - The notebooks use the python version as well as version of libraries shown in pyproject.toml file
- For the Python files (.py), try running them from your terminal or command prompt
- Don't worry if you get errors - we'll troubleshoot together in class

### 1.  **Goal 1: NLP Basics**: nlp-basics-2-transformers.ipynb
   - This is where we start with the fundamentals of natural language processing (NLP)
   - Topics covered:
     - Counting words in text (the simplest form of text analysis)
     - Introduction to how modern LLMs (AI) models process language (transformers, embeddings etc)

### 2. **Goal 2: Build an LLM from Scratch**: transformer-scratch.ipynb
   - Here we build the core pieces of an AI model ourselves
   - Topics covered:
     - Creating a simple word dictionary (vocabulary)
     - Breaking text into pieces the model can work with
     - Building the "attention" mechanism (how the model decides which words are important)
     - Step-by-step construction of transformer components
   - Note: Code derived from github created by Sebastian Raschka
     - Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
   - Python file used in this notebook: **gpt-dummy.py**
     - A complete mini-version of GPT that demonstrates transformers architecture

### 3. **Goal 3: Production Code**: nlp-api-production-code.ipynb
   - This shows how to use NLP and AI models in real-world applications
   - Topics covered:
     - Organizing code using classes (object-oriented programming)
     - Making your code reusable and maintainable
     - Practical patterns for working with AI in production
   - What to do: Look at how the code is organized into classes and methods
   - Python file used in this notebook: **llm_inference_runner.py**
     - A tool for running AI models on multiple texts at once with progress tracking
     - Shows how to connect to different AI services and handle errors in production
     - Read the code to see how batch processing works

---

- Useful tools I will show how to use in class that
    - [uv python and library manager](https://docs.astral.sh/uv/getting-started/installation/)
        - `uv sync` - Install dependencies from `pyproject.toml` file in the root folder
        - `uv run python -m src/gpt-dummy` - Run Python file using uv (helps with version contorl)
    - [dvc](https://dvc.org/doc/install) 
        - Version control for large data files and datasets
        - Sync data to/from remote storage (S3, GCS, etc.)
    - [weights and biases](https://docs.wandb.ai/quickstart)
        - Track and visualize machine learning experiments
        - Log metrics, artifacts, and compare model runs