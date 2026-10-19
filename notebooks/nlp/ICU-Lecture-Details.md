# Lecture Files Overview

## What This Lecture is About

## Keywords
- LLM (Large Language Model): 大規模言語モデル 
- Dimensions: 次元 
- Dimension Reduction: 次元削減
- Natural Language Processing (NLP): 自然言語処理 
- Transformers: トランスフォーマー
- Supervised Learning: 教師あり学習
- Unsupervised Learning: 教師なし学習
- Tokenization: トークン化
- Embeddings: 埋め込み
- Neural Networks: ニューラルネットワーク
- Attention Mechanism: アテンション機構
- Pre-training: 事前学習
- Fine-tuning: ファインチューニング
- Training Data: 訓練データ
- Model Deployment: モデルデプロイ

### Goals
この講義では、自然言語処理（NLP）の基礎から高度なLLMまでを学びます。
目標1：NLPの基礎 - テキストを処理・分析するための基本技術（「AI」モデルを含む）を学びます。
目標2：LLMをゼロから構築する - 現代の大規模言語モデル（LLM）が実際にどのように機能するかを、自ら構築することによって理解します。
目標3：本番コード - 実世界のアプリケーションでNLPシステムを開発・デプロイするためのベストプラクティスを学びます。



## Notebooks explained

## 1. 目標1：NLPの基礎: `nlp-basics-2-transformers.ipynb`

* ここでは、自然言語処理（NLP）の基礎から始めます。
* **学習内容:**
    * テキスト内の単語カウント（最も単純なテキスト分析）
    * 現代のLLM（AI）モデルが言語を処理する方法の紹介（トランスフォーマー、埋め込みなど）

## 2. 目標2：LLMをゼロから構築する: `transformer-scratch.ipynb`

* ここでは、AIモデルの中核部分を自分たちで構築します。
* **学習内容:**
    * 単純な単語辞書（語彙）の作成
    * モデルが扱えるようにテキストを分割する
    * 「アテンション」メカニズム（モデルがどの単語が重要かを判断する方法）の構築
    * トランスフォーマーのコンポーネントの段階的な構築
* **注:** このコードはSebastian Raschka氏が作成したgithubを基にしています。
    * Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.
* **このノートブックで使用されるPythonファイル:** `gpt-dummy.py`
    * トランスフォーマーのアーキテクチャを示す、GPTの完全なミニバージョンです。

## 3. 目標3：本番コード: `nlp-api-production-code.ipynb`

* ここでは、NLPとAIモデルを実世界のアプリケーションで使用する方法を示します。
* **学習内容:**
    * クラスを使用したコードの整理（オブジェクト指向プログラミング）
    * コードを再利用可能で保守しやすくする方法
    * 本番環境でAIを扱うための実践的なパターン
* **学習のポイント:** コードがどのようにクラスとメソッドに整理されているかに注目してください。
* **このノートブックで使用されるPythonファイル:** `llm_inference_runner.py`
    * S複数のテキストに対して一度にAIモデルを実行し、進捗を追跡するためのツールです。
    * さまざまなAIサービスに接続し、本番環境でのエラーを処理する方法を示します。
    * コードを読んで、バッチ処理がどのように機能するかを確認してください。





### Goals
- LLM (Large Language Model): 大規模言語モデル 
- Dimensions: 次元 
- Dimension Reduction: 次元削減
- Natural Language Processing (NLP): 自然言語処理 
- Transformers: トランスフォーマー 
- This lecture teaches Natural Language Processing (NLP) from basics to advanced LLMs
- **Goal 1: NLP Basics** - Learn fundamental techniques for processing and analyzing text (including "AI" models)
- **Goal 2: Build an LLM from Scratch** - Understand how modern large language models (LLMs) actually work by building one
- **Goal 3: Production Code** - Study best practices to developdeploy NLP systems in real-world applications

## Notebooks explained
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


**Please review the following notebooks and try running the code BEFORE the lecture:**

## A. Helpful tools to run the code below

-  Make sure you have Python and Jupyter installed on your computer
    - Python version and library versions can be viewed in`pyproject.toml`
        - Use [uv python and library manager](https://docs.astral.sh/uv/getting-started/installation/) (I will show how to use this in class)
- Open the notebook files (.ipynb) in Jupyter and try running them cell by cell
    - The notebooks use the python version as well as version of libraries shown in pyproject.toml file
- For the Python files (.py), try running them from your terminal or command prompt
- Don't worry if you get errors - we'll troubleshoot together in class


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
