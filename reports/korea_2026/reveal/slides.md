<!-- .slide: style="text-align: center;" -->
# AI Agents, Verification, and the Future of Research

<br>

### Cory Baird
Assistant Professor, University of Tokyo

<br>

<p style="color: #8c959f; font-size: 0.85em;">Korea 2026</p>

Note:
Welcome audience of policymakers, economists, and finance leaders.

---

## Overview: AI Agents, Verification and the Future of Research

<div class="lead-card" style="font-size: 1.25em; text-align: center; padding: 24px;">
AI supercharged has supercharge research and work output
</div>

<div style="margin-top: 36px; text-align: center;">
<p class="fragment" style="font-size: 1.3em; font-weight: 600; color: var(--brand-navy); margin-bottom: 22px;">How can we verify?</p>
<p class="fragment" style="font-size: 1.3em; font-weight: 600; color: var(--brand-teal); margin-bottom: 22px;">How can we trust?</p>
<p class="fragment" style="font-size: 1.3em; font-weight: 600; color: #854d0e;">Do we need new systems for work, education, and research?</p>
</div>

Note:
Reveal each question sequentially.
Three core themes for today.

---

## Early AI Earthquake: Seoul 2016

<div class="grid-60-40">
<div>
<iframe width="100%" height="400" 
  src="https://www.youtube-nocookie.com/embed/8tq1C8spV_g?start=22" 
  title="AlphaGo Trailer" 
  frameborder="0" 
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
  referrerpolicy="strict-origin-when-cross-origin" 
  allowfullscreen 
  style="border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
</iframe>
<p style="font-size: 0.72em; text-align: center; margin-top: 8px;">
  <a href="https://www.youtube.com/watch?v=8tq1C8spV_g&t=22s" target="_blank" rel="noopener noreferrer">Link: AlphaGo Documentary Trailer</a>
</p>
</div>
<div>
<h3>AlphaGo vs. Lee Sedol</h3>
<ul>
  <li>DeepMind's AlphaGo defeated Go champion Lee Sedol in Seoul</li>
  <li>Proved power of "AI" (neural nets)</li>
</ul>
</div>
</div>

Note:
Anchor the talk in Korea's pivotal historical role in modern AI history.

---

## AI $\neq$ new technology

<div class="timeline-container">
<div class="timeline-grid">

  <!-- Column 1: 1956 (Top) -->
  <div class="timeline-item-top" style="grid-column: 1;">
    <div class="timeline-detail">AI term coined<br>Symbolic foundations</div>
    <div class="timeline-name">Dartmouth</div>
    <div class="timeline-year">1956</div>
  </div>

  <!-- Column 2: 1982 (Bottom) -->
  <div class="timeline-item-bottom" style="grid-column: 2;">
    <div class="timeline-year">1982</div>
    <div class="timeline-name">Hopfield Net</div>
    <div class="timeline-detail">Recurrent neural memory<br>AI winter thaw</div>
  </div>

  <!-- Column 3: 1997 (Top) -->
  <div class="timeline-item-top" style="grid-column: 3;">
    <div class="timeline-detail">Defeats chess champion<br>Heuristic tree search</div>
    <div class="timeline-name">Deep Blue</div>
    <div class="timeline-year">1997</div>
  </div>

  <!-- Column 4: 2012 (Bottom) -->
  <div class="timeline-item-bottom" style="grid-column: 4;">
    <div class="timeline-year">2012</div>
    <div class="timeline-name">AlexNet</div>
    <div class="timeline-detail">GPU deep learning boom<br>ImageNet error halved</div>
  </div>

  <!-- Column 5: 2015 (Top) -->
  <div class="timeline-item-top" style="grid-column: 5;">
    <div class="timeline-detail">Speech &amp; translation leap<br>Multi-layer scaling</div>
    <div class="timeline-name">Deep Learning</div>
    <div class="timeline-year">2015</div>
  </div>

  <!-- Column 6: 2017 (Bottom) -->
  <div class="timeline-item-bottom" style="grid-column: 6;">
    <div class="timeline-year">2017</div>
    <div class="timeline-name">Transformer</div>
    <div class="timeline-detail">Attention mechanisms<br>Foundation for modern LLMs</div>
  </div>

  <!-- Column 7: 2022 (Top) -->
  <div class="timeline-item-top" style="grid-column: 7;">
    <div class="timeline-detail">Generative foundation models<br>Mass public adoption</div>
    <div class="timeline-name">ChatGPT</div>
    <div class="timeline-year">2022</div>
  </div>

  <!-- Central Axis & Nodes -->
  <div class="timeline-axis-line"></div>
  <div class="timeline-dot" style="grid-column: 1;"></div>
  <div class="timeline-dot" style="grid-column: 2;"></div>
  <div class="timeline-dot" style="grid-column: 3;"></div>
  <div class="timeline-dot" style="grid-column: 4;"></div>
  <div class="timeline-dot" style="grid-column: 5;"></div>
  <div class="timeline-dot" style="grid-column: 6;"></div>
  <div class="timeline-dot" style="grid-column: 7;"></div>

</div>
</div>

Note:
Contextualize Seoul 2016 within seven decades of algorithmic progress
Generative systems build on long evolutionary trajectory

---

## 1936--Not a new question: Are machines intelligent?

<div class="grid-60-40">
<div>
<iframe width="100%" height="400" 
  src="https://www.youtube-nocookie.com/embed/lTMEUA6VnJE" 
  title="The Imitation Game Scene" 
  frameborder="0" 
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
  referrerpolicy="strict-origin-when-cross-origin" 
  allowfullscreen 
  style="border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
</iframe>
<p style="font-size: 0.72em; text-align: center; margin-top: 8px;">
  <a href="https://www.youtube.com/watch?v=lTMEUA6VnJE" target="_blank" rel="noopener noreferrer">Link: The Imitation Game (Interrogation Scene)</a>
</p>
</div>
<div>
<h3>Inventor of Computer, Alan Turing, questioned: Can machines think?</h3>
<ul>
  <li>"It is possible to invent a single machine which can be used to compute any computable sequence." (1936) </li>
  <li>"If a machine is expected to be infallible, it cannot also be intelligent." (1947)</li>
</ul>
</div>
</div>

Note:
Turing shifted question from biological essence to functional capability
Core benchmark for evaluating machine output

---

## 1958--Not a new question: Are machines intelligent?

<div class="grid-2" style="align-items: center;">
<div style="text-align: center;">
  <img src="figures/nyt-1958.jpeg" alt="New York Times 1958 Perceptron Article" style="max-height: 480px; width: auto; border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.15);">
  <p style="font-size: 0.72em; text-align: center; margin-top: 8px; color: #57606a;">New York Times (July 8, 1958)</p>
</div>
<div>
<h3>An early version of AI (perceptron)</h3>
<ul>
  <li>Frank Rosenblatt invented perceptron</li>
  <li>Predicted it would be able to: "perceive, recognize and identify its surroundings"</li>
</ul>
</div>
</div>

Note:
Early perceptron demonstrations sparked massive media predictions
Parallels modern claims around emergent intelligence

---

## 2020s--Not a new question: Are machines intelligent?

<div class="grid-3" style="align-items: start; margin-top: 10px;">
<div style="text-align: center;">
  <iframe width="100%" height="420" 
    src="https://www.youtube-nocookie.com/embed/Hw7_f_NwmX0" 
    title="Dario Amodei (Anthropic)" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
    referrerpolicy="strict-origin-when-cross-origin" 
    allowfullscreen 
    style="max-width: 250px; margin: 0 auto; display: block; border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
  </iframe>
  <p style="font-size: 0.72em; margin-top: 8px; font-weight: 600; color: var(--brand-navy);">
    <a href="https://www.youtube.com/shorts/Hw7_f_NwmX0" target="_blank" rel="noopener noreferrer">Dario Amodei (Anthropic)</a>
  </p>
</div>
<div style="text-align: center;">
  <iframe width="100%" height="420" 
    src="https://www.youtube-nocookie.com/embed/kPzeqQoshx4" 
    title="Sam Altman (OpenAI)" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
    referrerpolicy="strict-origin-when-cross-origin" 
    allowfullscreen 
    style="max-width: 250px; margin: 0 auto; display: block; border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
  </iframe>
  <p style="font-size: 0.72em; margin-top: 8px; font-weight: 600; color: var(--brand-navy);">
    <a href="https://www.youtube.com/shorts/kPzeqQoshx4" target="_blank" rel="noopener noreferrer">Sam Altman (OpenAI)</a>
  </p>
</div>
<div style="text-align: center;">
  <iframe width="100%" height="420" 
    src="https://www.youtube-nocookie.com/embed/UktpwiM0MeI" 
    title="Demis Hassabis (Google DeepMind)" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
    referrerpolicy="strict-origin-when-cross-origin" 
    allowfullscreen 
    style="max-width: 250px; margin: 0 auto; display: block; border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
  </iframe>
  <p style="font-size: 0.72em; margin-top: 8px; font-weight: 600; color: var(--brand-navy);">
    <a href="https://www.youtube.com/shorts/UktpwiM0MeI" target="_blank" rel="noopener noreferrer">Demis Hassabis (DeepMind)</a>
  </p>
</div>
</div>

Note:
Frontier lab leaders debate AGI timelines and machine capability
Recurring question on whether scale yields genuine intelligence

---

<!-- .slide: class="section-break-slide" -->
# Part 1: Demystifying Modern AI

### LLMs: Are they smart or are we dumb?

---

## The word "AI" = Zero Meaning

### The most important distinction: Generative vs. Predictive

<div class="grid-2">
<div class="card card-teal">
<h3>Generative AI</h3>
<p style="font-size: 0.82em; font-weight: 600; color: var(--brand-teal); margin-bottom: 10px;">Generating content</p>
<ul>
  <li><strong>Mechanism:</strong> Model underlying data distributions to produce novel samples</li>
  <li><strong>Objectives:</strong> Text generation, summarization, conversational agents</li>
  <li><strong>Outputs:</strong> Natural language, source code, synthetic data</li>
  <li><strong>Examples:</strong> ChatGPT (text), Copilot (code completion)</li>
</ul>
</div>
<div class="card">
<h3>Predictive AI</h3>
<p style="font-size: 0.82em; font-weight: 600; color: var(--brand-navy); margin-bottom: 10px;">Answering a question</p>
<ul>
  <li><strong>Mechanism:</strong> Analyze historical patterns to identify unknown values or categories</li>
  <li><strong>Objectives:</strong> Classification, numerical regression, anomaly detection</li>
  <li><strong>Outputs:</strong> Continuous values, probabilities, labels (Yes/No)</li>
  <li><strong>Examples:</strong> Revenue forecasts, fraud detection, risk scoring</li>
</ul>
</div>
</div>


---

## When people say AI today: Almost always = Generative
### Generative AI: Almost always = Transformers

<ul style="margin-bottom: 12px;">
  <li>Foundational architecture from landmark "Attention Is All You Need" paper (2017)</li>
  <li>All major models same share architecture</li>
</ul>

<iframe src="https://sebastianraschka.com/llm-architecture-gallery/#architecture-gallery" width="100%" height="410" frameborder="0" style="border: 1px solid var(--border-color); border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.12); pointer-events: auto;" allow="fullscreen"></iframe>

<p style="font-size: 0.72em; text-align: center; margin-top: 6px;">
  <a href="https://sebastianraschka.com/llm-architecture-gallery/#architecture-gallery" target="_blank" rel="noopener noreferrer">Link: Sebastian Raschka's LLM Architecture Gallery</a>
</p>

Note:
Interactive gallery catalogs transformer architectural variants across frontier LLMs
All modern large models derive from the original 2017 attention mechanism

---

## Visualizing the Transformer

<div class="grid-60-40">
<div style="text-align: center;">
<img src="figures/transformer_self_attention_map.png" alt="Self-Attention Map" style="max-height: 380px; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
<div style="margin-top: 10px;">
  <a href="https://ig.ft.com/generative-ai/" target="_blank" rel="noopener noreferrer" class="btn-action">
    Launch Live FT Visual Guide &rarr;
  </a>
</div>
</div>
<div>
<h3>Next-Token Mechanics</h3>
<ul>
  <li>Text mapped to high-dimensional token vectors</li>
  <li><strong>Self-Attention:</strong> Weighs relationships across prior context</li>
  <li>Predicts mathematically plausible continuations</li>
</ul>
<h3 style="margin-top: 20px;">Key Takeaway</h3>
<ul>
  <li>Pattern continuation without empirical ground truth</li>
  <li>Must be constrained when handling financial data</li>
</ul>
</div>
</div>

---

## Stochastic Parrots...???

<div class="grid-2">
<div class="card">
<h3>LLMs: A Normal Technology</h3>
<p style="font-size: 0.82em; font-weight: 600; color: var(--brand-navy); margin-bottom: 10px;">The Engineering Tool Perspective</p>
<ul>
  <li><strong>Core Mechanism:</strong> Statistical next-token prediction over text distributions</li>
  <li><strong>Internal State:</strong> Surface linguistic correlations without grounded meaning</li>
  <li><strong>Failure Mode:</strong> Confident hallucinations and brittle multi-step reasoning</li>
  <li><strong>Practical Role:</strong> Useful cognitive artifact requiring strict verification harnesses</li>
</ul>
</div>
<div class="card card-teal">
<h3>LLMs: A Machine God</h3>
<p style="font-size: 0.82em; font-weight: 600; color: var(--brand-teal); margin-bottom: 10px;">The Superintelligence Perspective</p>
<ul>
  <li><strong>Core Claim:</strong> Scale yields genuine emergent understanding and artificial general intelligence</li>
  <li><strong>Internal State:</strong> Autonomous reasoning entities forming beliefs and intentionality</li>
  <li><strong>Failure Mode:</strong> Anthropomorphic projection mistaking syntactic fluency for consciousness</li>
  <li><strong>Practical Risk:</strong> Uncritical delegation of policy and finance decisions to black-box models</li>
</ul>
</div>
</div>

Note:
Stochastic parrots debate pits statistical pattern matching against claims of emergent cognition
Reality in research: powerful workflow accelerator devoid of autonomous understanding

---

## Even "Dumb" AI Is Revolutionary

<div class="lead-card">
<strong>Utility Without AGI:</strong> Immense economic value does not require artificial general intelligence. Automating repetitive data plumbing transforms white-collar research.
</div>

<div class="grid-2">
<div>
<h3>What AI Accelerates</h3>
<ul>
  <li>Parsing messy unstructured regulatory filings</li>
  <li>Extracting sentiment from central bank transcripts</li>
  <li>Standardizing disparate cross-country data tables</li>
</ul>
</div>
<div>
<h3>The Strategic Division</h3>
<ul>
  <li><strong>Humans:</strong> Institutional reality checks and causal design</li>
  <li><strong>Agents:</strong> Rigidly constrained workflow acceleration</li>
</ul>
</div>
</div>

---

<!-- .slide: class="section-break-slide" -->
# Part 2: The Verification Crisis

### Has AI broken scientific validation, or just exposed it?

---

## The Generation vs Verification Asymmetry

<div class="quote-card">
"AI has collapsed the cost of doing certain kinds of work: coding, data construction, empirical analysis... the cost of verifying that the work is correct has moved much less."<br>
<span style="font-style: normal; font-weight: 600; font-size: 0.85em; color: var(--brand-navy); font-family: var(--r-main-font);">&mdash; Paul Goldsmith-Pinkham (Yale SOM)</span>
</div>

<div class="grid-2">
<div class="card">
<h3>Generation Cost: Zero</h3>
<ul>
  <li>Instant regression scripts and formatted exhibits</li>
  <li>Plausible narrative memos generated on demand</li>
  <li>Mathematical errors remain invisible to surface checks</li>
</ul>
</div>
<div class="card card-teal">
<h3>Verification Cost: Constant</h3>
<ul>
  <li>Auditing complex merges requires deep domain expertise</li>
  <li>Human review cannot scale to thousands of machine runs</li>
  <li>Solution: Automate verification, not just drafting</li>
</ul>
</div>
</div>

---

## Did the Deficit Start with AI?

<div class="grid-2">
<div>
<h3>Pre-AI Economics Was Fragile</h3>
<ul>
  <li>Replication crises existed long before transformers</li>
  <li>Ad-hoc scripts lacking reproducible environments</li>
  <li>Inconsistent folder layouts and undocumented joins</li>
  <li>Fragile spreadsheets harboring silent calculation bugs</li>
</ul>
</div>
<div>
<h3>AI Removed the Volume Brake</h3>
<ul>
  <li>Human typing speed previously acted as a rate limiter</li>
  <li>Low volume kept the verification deficit hidden</li>
  <li>AI unleashed infinite volume without fixing auditing</li>
  <li><strong>Core Reality:</strong> Verification has always been the bottleneck</li>
</ul>
</div>
</div>

---

## Why Stakes Are High in Policy &amp; Finance

<div class="grid-3">
<div class="card">
<h3>Central Banking</h3>
<ul>
  <li>Policy rates depend on exact inflation dynamics</li>
  <li>Date misalignment distorts monetary decisions</li>
  <li>Requires deterministic, auditable models</li>
</ul>
</div>
<div class="card">
<h3>Public Finance</h3>
<ul>
  <li>Tax forecasts govern multi-billion national budgets</li>
  <li>Subtle parameter errors cause massive fiscal gaps</li>
  <li>Demands reproducible empirical evidence</li>
</ul>
</div>
<div class="card card-teal">
<h3>Sovereign Risk</h3>
<ul>
  <li>Risk indices drive bond yields and capital flows</li>
  <li>Black-box scores trigger intellectual herding</li>
  <li>Regulators require verified data provenance</li>
</ul>
</div>
</div>

---

<!-- .slide: class="section-break-slide" -->
# Part 3: Infrastructure Over Prompts

### Hard constraints beat fragile prompt engineering

---

## The Prompting Trap

<div class="grid-2">
<div>
<h3>Prompts Are Fragile</h3>
<ul>
  <li>"Act as an econometrician" is non-deterministic</li>
  <li>Identical prompts generate different code across runs</li>
  <li>Checks rhetorical style, not mathematical logic</li>
  <li>Chat reviews reinforce the model's own biases</li>
</ul>
</div>
<div>
<h3>Errors Live in the Plumbing</h3>
<ul>
  <li>Real mistakes happen in low-level data plumbing:
    <ul>
      <li>Mismatched frequency resampling</li>
      <li>Silent dropping of balanced panel observations</li>
      <li>Unlogged variable transformations</li>
    </ul>
  </li>
  <li>Linguistic prompts cannot fix deterministic plumbing</li>
</ul>
</div>
</div>

---

## The Macro Agent Harness Concept

<div class="lead-card">
<strong>Harness Engineering:</strong> Replace unconstrained code drafting with rigid software contracts. Agents never write econometric engines from scratch; they modify declared YAML configs.
</div>

<div class="grid-2">
<div class="card">
<h3>Unconstrained (Fragile)</h3>
<ul>
  <li>Writes bespoke scripts from scratch</li>
  <li>Executes ad-hoc transformations in memory</li>
  <li>Zero reproducibility guarantees</li>
</ul>
</div>
<div class="card card-teal">
<h3>Harness-Constrained (Verifiable)</h3>
<ul>
  <li>Calls pre-built, tested domain libraries</li>
  <li>Modifies parameters within strict schemas</li>
  <li>Emits deterministic execution manifests</li>
</ul>
</div>
</div>

---

## The Open-Source Research Stack

<div class="grid-2">
<div>
<div class="card" style="margin-bottom: 15px;">
<h3>1. Database: OpenMacro</h3>
<p style="font-size: 0.85em; margin: 0;">11.4M observations across 150+ datasets. Harmonizes World Bank, IMF, BIS data across frequencies.</p>
</div>
<div class="card">
<h3>2. Econometrics: Stats Transformer</h3>
<p style="font-size: 0.85em; margin: 0;">Declarative YAML pipelines. Estimators benchmarked against STATA, MATLAB, and R.</p>
</div>
</div>
<div>
<div class="card card-teal" style="margin-bottom: 15px;">
<h3>3. NLP: Auto-Econ Sentiment</h3>
<p style="font-size: 0.85em; margin: 0;">Replaces black-box LLMs with auditable macro dictionaries and open transformer models.</p>
</div>
<div class="card card-teal">
<h3>4. Theory &amp; Pipeline: DSGE Runner</h3>
<p style="font-size: 0.85em; margin: 0;">Industrial DSGE modeling 10x faster than Dynare with deterministic artifact caching.</p>
</div>
</div>
</div>

---

<!-- .slide: class="section-break-slide" -->
# Part 4: Safe Agent Architecture

### Sandboxed, vendor-neutral, and tool-routed systems

---

## The Three Architecture Guardrails

<div class="grid-3">
<div class="card">
<h3>1. Role Subagents</h3>
<ul>
  <li>Single agents suffer from self-validating hallucinations</li>
  <li>Separate into narrow roles:
    <ul>
      <li><strong>Data Agent:</strong> DB access only</li>
      <li><strong>Model Agent:</strong> Config only</li>
      <li><strong>Reviewer:</strong> Read-only</li>
    </ul>
  </li>
</ul>
</div>
<div class="card">
<h3>2. Sandboxing</h3>
<ul>
  <li>General AI has dangerous filesystem access</li>
  <li>Enforce strict OS isolation:
    <ul>
      <li>Raw data mounted read-only</li>
      <li>Isolated scratch spaces</li>
      <li>Core code immutable</li>
    </ul>
  </li>
</ul>
</div>
<div class="card card-teal">
<h3>3. Vendor Neutrality</h3>
<ul>
  <li>Proprietary lock-in creates major vulnerabilities:
    <ul>
      <li>Silent model drift</li>
      <li>Pricing monopoly</li>
      <li>Intellectual herding</li>
    </ul>
  </li>
  <li>Modular harness supporting open weights</li>
</ul>
</div>
</div>

---

## Single Agent vs Multi-Agent Harness

<div class="grid-2">
<div>
<h3>Vulnerable: Single Agent</h3>
<pre class="diagram-box">
     [User Request]
            │
            ▼
┌───────────────────────────┐
│    Unbounded General AI   │
│  - Writes code            │
│  - Downloads data         │
│  - Audits own output      │
└───────────────────────────┘
            │
            ▼
[High Hallucination Risk]
[Data Overwrite Exposure]
</pre>
</div>
<div>
<h3>Robust: Multi-Agent Harness</h3>
<pre class="diagram-box">
         [User Request]
                │
                ▼
      [Orchestrator Agent]
      ┌─────────┴─────────┐
      ▼                   ▼
[Data Agent]        [Model Agent]
(OpenMacro DB)     (Stats Engine)
      │                   │
      └─────────┬─────────┘
                ▼
        [Reviewer Agent]
        (Read-Only Audit)
</pre>
</div>
</div>

---

## The Golden Rule: Read-Only Data

<div class="lead-card">
<strong>First Rule of Research MLOps:</strong> Never allow an AI agent write-access to raw data. Raw observational data must remain an immutable foundation.
</div>

<div class="grid-2">
<div>
<h3>Common Silent Failures</h3>
<ul>
  <li>Agent silently drops outliers without logging</li>
  <li>Column headers overwritten during merges</li>
  <li>Temporary patches lost when container exits</li>
</ul>
</div>
<div>
<h3>The Sandbox Protection</h3>
<ul>
  <li><strong>Read-Only Mounts:</strong> OS blocks modifications</li>
  <li><strong>Scratch Folders:</strong> Isolated intermediate storage</li>
  <li><strong>Cryptographic Hashes:</strong> Bitwise verification</li>
</ul>
</div>
</div>

---

<!-- .slide: class="section-break-slide" -->
# Part 5: The Future of Research

### Essential human skills and policy autonomy

---

## Human Capital in the Agentic Era

<div class="grid-2">
<div class="card">
<h3>Skills Depreciating</h3>
<ul>
  <li>Memorizing coding syntax and commands</li>
  <li>Manual spreadsheet data entry</li>
  <li>Boilerplate slide and document formatting</li>
  <li>Basic literature search without context</li>
</ul>
</div>
<div class="card card-teal">
<h3>Skills Appreciating</h3>
<ul>
  <li><strong>Data Literacy:</strong> Evaluating provenance and bias</li>
  <li><strong>Causal Identification:</strong> Designing clean empirical tests</li>
  <li><strong>Domain Reality Checks:</strong> Catching subtle institutional errors</li>
  <li><strong>Cross-Disciplinary Translation:</strong> Framing policy decisions</li>
</ul>
</div>
</div>

---

## Sovereign AI &amp; Policy Autonomy

<div class="grid-2">
<div>
<h3>The Concentration Risk</h3>
<ul>
  <li>Frontier models concentrated in the US and China</li>
  <li>Closed APIs carry commercial and political vulnerability</li>
  <li>Vendors silently alter weights without notice</li>
</ul>
</div>
<div>
<h3>The Open Science Imperative</h3>
<ul>
  <li>Central banks and treasuries require auditable foundations</li>
  <li>Deploying open weights preserves institutional confidentiality</li>
  <li>Public policy must remain independently verifiable</li>
</ul>
</div>
</div>

---

## Core Takeaways

<div class="lead-card">
<strong>Strategic Summary:</strong> AI does not have to cause a verification crisis. Replace conversational prompting with verified domain harnesses to gain massive efficiency while safeguarding rigor.
</div>

<div class="grid-3">
<div class="card">
<h3>1. Build the Harness</h3>
<p style="font-size: 0.88em; margin: 0;">Invest in deterministic domain libraries, typed schemas, and automated validation.</p>
</div>
<div class="card">
<h3>2. Isolate &amp; Sandbox</h3>
<p style="font-size: 0.88em; margin: 0;">Enforce role separation and protect raw observational data with read-only mounts.</p>
</div>
<div class="card card-teal">
<h3>3. Elevate Human Judgment</h3>
<p style="font-size: 0.88em; margin: 0;">Shift human expertise toward data provenance, causal design, and policy trade-offs.</p>
</div>
</div>

---

<!-- .slide: style="text-align: center;" -->
# Thank You

### Questions &amp; Discussion

**Cory Baird**  
Assistant Professor, Markets + Governance, University of Tokyo  
[baird@g.ecc.u-tokyo.ac.jp](mailto:baird@g.ecc.u-tokyo.ac.jp)
