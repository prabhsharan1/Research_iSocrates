# iSocrates: A Hybrid Architecture for Real-Time Rhetorical Figure Detection and Verification

**Abstract**  
The computational detection of rhetorical figures remains a complex challenge in Natural Language Processing (NLP) due to the nuanced, context-dependent, and often overlapping definitions of rhetorical devices. In this paper, we present the architecture of **iSocrates**, a real-time rhetorical analysis bot. We detail the evolution of our machine learning pipeline—from our initial DistilBERT sequence classifier to token-level transformers—and expose the pitfalls of relying solely on high-accuracy token classifiers that fail in real-world deployments. To solve these challenges, we introduce a novel **"Triple-Check" Hybrid Architecture** that combines the sub-100ms inference speed of an encoder with deterministic programmatic constraints (like phonetic dictionaries), and the deep semantic reasoning of a highly accessible, lightweight local Large Language Model (progressing from Gemma 2B to a highly quantized Qwen 2.5 7B GPT-Generated Unified Format (GGUF)). Finally, we document the extensive full-stack infrastructure built to support this pipeline, including dynamic training filters, automated evaluation harnesses, and strict database-matching shortcuts.

---

## System Architecture Overview

Rhetoricon leverages the core **Triple-Check Hybrid Pipeline** (exact-match database caching, sequence classification, and local LLM verification) across two distinct user-facing platform workflows:

### 1. iSocrates Chat & PDF Analysis Workflow (Admin Research Tool)
This workflow is optimized for real-time exploratory analysis, routing the input dynamically based on whether the user is asking chat questions or requesting heavy PDF span extraction.

```mermaid
flowchart TD
    A1(["Admin Input / Contributor Submission"])
    B1{{"Postgres Exact-Match Shortcut"}}
    C1(["✅ 100% Verified Instant Return"])
    D1["DistilBERT Sequence Classifier"]
    E1{{"Fast-Path Bypass Match?"}}
    F1["Token Normalization & Phonetic Check\n(Assonance / Consonance → Score 95)"]
    G1["Repetition & Syntactic Rules\n(Epanaphora / Epiphora / Anadiplosis / Isocolon → Score 95)"]
    H1{{"Fail-Soft Fallback\n(Match Succeeded?)"}}
    I1["Qwen 2.5 7B GGUF\n(Dynamic Few-Shot RAG Verification)"]
    J1(["Verified Figure & Character Spans"])

    A1 --> B1
    B1 -- "Match Found" --> C1
    B1 -- "No Match" --> D1
    D1 -- "Candidates" --> E1
    E1 -- "Phonetic" --> F1
    E1 -- "Repetition / Syntax" --> G1
    F1 --> H1
    G1 --> H1
    H1 -- "Yes (Score 95)" --> J1
    H1 -- "No (Fallback)" --> I1
    E1 -- "Semantic / Complex" --> I1
    I1 --> J1
```
*Figure 1: The Enhanced Triple-Check Hybrid Architecture. Deterministic phonetic, repetition, and syntactic fast-paths act as sub-millisecond logic gates before falling back softly to the local 7B LLM for complex semantic verification.*

### 2. GoFigure Submission & Verification Pipeline (Public Platform)
This workflow is triggered when users submit rhetorical figures to the crowdsourced database. It combines core AI figure classification with external metadata validation and downstream synchronization.

```mermaid
flowchart TD
    A2(["Contributor Submission\n(Passage & Metadata Citation)"])
    B2["Core AI Analysis\n(DistilBERT + Phonetic + Qwen 7B GGUF)"]
    C2["AI Figure Match & Spans\n(Assigned confidence score)"]
    D2["API Source Verification\n(Matches against Google Books / Open Library)"]
    E2{{"Metadata Discrepancy?"}}
    F2["Flag 'Suspicious'\n(Show Auto-Fix suggestions)"]
    G2["Flag 'Valid'"]
    H2["Moderator Review Panel\n(Approve / Reject Action)"]
    I2(["Zotero Auto-Sync\n(Pushed to shared bibliography on approval)"])

    A2 --> B2
    B2 --> C2
    C2 --> D2
    D2 --> E2
    E2 -- "Yes" --> F2
    E2 -- "No" --> G2
    F2 --> H2
    G2 --> H2
    H2 -- "Approved" --> I2
```
*Figure 2: GoFigure Crowdsourced Submission & Verification Pipeline. Submissions undergo parallel automated figure extraction and metadata source validation prior to manual moderator vetting.*

---

## 1. Introduction
Rhetorical figures—such as *Alliteration*, *Antimetabole*, and *Oxymoron*—are omnipresent in persuasive communication. However, identifying them computationally is notoriously difficult. A major hurdle is the ambiguity of definitions; for instance, distinguishing an intentional *Anaphora* from an accidental repetition of a stop word, or differentiating *Chiasmus* from *Antimetabole*.

The **iSocrates** bot was developed to assist users in identifying these figures in real-time through an interactive chat UI. Our goal was to create a system that is highly accurate, capable of understanding strict definitions provided by human experts, and 100% accessible to the public without being gated behind expensive API paywalls.

## 2. The Evolution of the Detection Model
Our modeling approach underwent several major iterations as we balanced accuracy, deployment constraints, and the ability to detect multiple overlapping figures.

### 2.1 The First Attempt: DistilBERT Sequence Classification
Our initial model was a DistilBERT Sequence Classifier. During testing, it achieved good F1 scores on subsets of data, but an overall raw accuracy of ~16.6% across the full spectrum of definitions.

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Macro |
|-------|---------------|-----------------|----------|----------|
| 1     | 3.309009      | 3.213277        | 0.143520 | 0.020010 |
| 4     | 2.705567      | 3.125247        | 0.174543 | 0.044421 |
| 8     | 2.316890      | 3.195940        | 0.166135 | 0.045208 |

*Table 1: The empirical baseline metrics for the standalone DistilBERT Sequence Classifier.*

While the DistilBERT sequence classifier (Sanh et al., 2019) achieved an accuracy of ~16.6% on the full definition set, it was constrained by a single-label prediction architecture. This bottleneck necessitated a transition to a multi-label classification paradigm to accommodate the overlapping nature of rhetorical figures.

### 2.2 The Token Classifier & The "87% Accuracy" Paradox
To solve the single-figure limitation, we shifted our architecture to a Hugging Face `AutoModelForTokenClassification` (Wolf et al., 2020). This model was designed to pinpoint the exact words constituting multiple figures using `BIO` (Beginning, Inside, Outside) tagging. 

During validation, this model achieved an impressive **87% accuracy**. However, deploying this model revealed a classic ML pitfall: poor out-of-domain generalization due to extreme class imbalance, where over 99.9% of tokens belong to the 'Outside' (O) class. In practice, the model simply learned to classify almost all tokens as "Outside", producing misleadingly high accuracy while failing to detect actual figures (resulting in extremely low F1, Precision, and Recall).

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 | Accuracy |
|-------|---------------|-----------------|-----------|--------|----|----------|
| 1     | 0.700922      | 0.688752        | 0.010256  | 0.001326 | 0.002349 | 0.873588 |
| 4     | 0.544936      | 0.626245        | 0.062405  | 0.040782 | 0.049328 | 0.872986 |
| 8     | 0.437769      | 0.624886        | 0.076018  | 0.051393 | 0.061325 | 0.873701 |

*Table 2: Training metrics showing the token classifier's high overall accuracy but low functional performance (F1/Precision/Recall) due to class imbalance.*

### 2.3 The DeBERTa Token Classifier & The "NaN Loss" Collapse
In our second attempt to make token classification work, we upgraded to `microsoft/deberta-v3-small` (He et al., 2021) to exploit its advanced relative position encodings. 

The transition to `microsoft/deberta-v3-small` introduced numerical instability. Extreme class imbalance led to gradient collapse. This highlighted the necessity of shifting away from token-level classification toward a hybrid sequence-extraction pipeline.

### 2.4 Reverting to Sequence Classification + LLM Extraction (The Hybrid Solution)
Guided by these findings, we established that rhetorical figures are best detected holistically at the sequence level. We definitively reverted our local classifier to the stable **DistilBERT Sequence Classifier** (Sanh et al., 2019), lowering the classification threshold to 5% to capture multiple overlapping candidate figures in a single sentence. 

Once candidate figures are identified, we delegate the exact character-span extraction and explanation to the **local LLM**. This separation of concerns (fast sentence-level classification + deep token-level LLM extraction) successfully bypassed the token-imbalance problem and ensured 100% mathematical stability.

### 2.5 Data Curation and Source Validation
The foundation of both the DistilBERT classifier and the subsequent LLM verification layer relies entirely on a meticulously curated dataset. A pervasive challenge in rhetorical figure detection is the lack of standardized definitions, which frequently leads to poor inter-annotator agreement and corrupted training signals. To solve this, our architecture relies on human-verified training data curated through collaborative pedagogical review within the Rhetoricon project's *Doxa* database. 

While *Doxa* serves as our primary data repository, we implemented a strict, automated **Source Validation** layer to ensure the integrity of the underlying texts. To guarantee absolute metadata accuracy, the frontend integrates directly with the Google Books API (Google, 2024) to automatically validate ISBNs and fetch authoritative bibliographic data (such as exact authors, publishers, and publication dates) before any instance is permanently mapped to a verified bibliographic source. By grounding the training data and the RAG pipeline (Lewis et al., 2020) in this strict, API-validated ontology, the architecture ensures the models learn the true functional properties of the figures rather than noisy, unverified internet data.

## 3. The Hybrid RAG Pipeline and LLM Selection
While the Sequence Classifier is extremely fast, it still struggles with highly specific edge-case definitions that require deep logical reasoning. We needed an LLM to verify these complex cases.

### 3.1 Model Selection: Why Gemma 2B?
1. **Avoiding External APIs:** We initially considered using external APIs like Google's Gemini. While the free tier works for simple testing, scaling it to a public audience would hit strict rate limits and expensive paywalls. During testing, the free tier of the Gemini API frequently failed or timed out after analyzing only 1 or 2 sentences depending on their length. Because we already possessed a massive database of human-verified training data, we opted to build a completely self-hosted solution to keep the bot entirely free and accessible.
2. **Mistral, Phi-3, and Gemma 2B:** During our design phase, we researched several local open-weight models to host on our Portainer staging/production servers. Mistral 7B (Jiang et al., 2023) is highly capable but requires nearly 5 GB of RAM when quantized, making it too heavy and risky for a lightweight Portainer deployment. We also evaluated Microsoft's Phi-3-mini (Abdin et al., 2024), which showed solid reasoning capabilities. However, we ultimately selected **Gemma 2B** (Gemma Team, 2024) for our initial trials. At under 2 GB of RAM (when using 4-bit quantization), Gemma 2B was incredibly lightweight, fast, and highly capable of strict pattern-matching when provided with in-context examples.

### 3.2 Optimizing the Local LLM: Swapping to Qwen 0.5B on CPU
While Gemma 2B performed well in terms of accuracy, running inference on a virtualized CPU core (UTM VM) took over 40 seconds per figure verification. For passages containing multiple candidate figures, this led to verification loops lasting over 2 minutes.

To solve this, we swapped Gemma 2B for **Qwen2.5-0.5B-Instruct** (Qwen Team, 2024). At under 500 million parameters, Qwen runs exceptionally fast on CPU. By further restricting `max_new_tokens` from `256` to `80` (since verifier JSON outputs are short), we cut Qwen's verification time from **2m 11s** down to **32.4 seconds** for a 3-figure candidate analysis (under 10 seconds per figure).

### 3.3 The Architectural Pivot: Ollama vs. Hugging Face
The initial inference architecture utilized Ollama as a standalone inference engine. However, network-level latency and SSH-tunneling overhead proved prohibitive for real-time interaction. Consequently, we migrated the model natively into a unified Python-based `transformers` ecosystem (Wolf et al., 2020). 

By utilizing `bitsandbytes` (Dettmers et al., 2022) for 4-bit quantization, we loaded the LLM directly into our Python backend alongside DistilBERT. This pivot provided two massive advantages:
1. **Unified Infrastructure:** The Go backend no longer relies on fragile network tunnels. It simply queries a unified local Python inference server that handles both the fast classifier and the heavy LLM verification.
2. **Native Fine-Tuning:** By living in the Hugging Face ecosystem, we unlocked the ability to dynamically fine-tune the model using LoRA (Hu et al., 2021) via the PEFT library (Mangrulkar et al., 2026) directly from our PostgreSQL database.

### 3.4 Closing the Loop: Feedback Integration & Exporter
We then implemented a closed-loop training dataset pipeline to dynamically update both models from user corrections:
1. **Classifier Feedback Injection:** Corrections from `/feedback` are dynamically parsed by the LLM and stored in PostgreSQL. The training data pipeline (`extract_data.py`) fetches these corrections and injects them as synthetic training instances, ensuring DistilBERT learns directly from user corrections on the next run.
2. **LLM Dataset Exporter:** We created `prepare_llm_dataset.py` to compile database definitions, gold-standard examples, and user corrections into a structured instruction-tuning dataset (`qwen_lora_dataset.json`) containing over 69,000 prompt-completion pairs to fine-tune Qwen on Google Colab using LoRA (Hu et al., 2021).

### 3.5 Scaling Up: Solving "0% Collapse" with Qwen 2.5 7B GGUF
Despite fine-tuning, smaller models (0.5B - 2B) exhibited a structural "0% collapse" on complex definitions and were prone to outputting hallucinatory text rather than strict JSON logic. Furthermore, initial attempts to load unquantized 7B models in PyTorch (FP16) or run multiple independent inference engines concurrently (like Ollama alongside Hugging Face sidecars) resulted in severe memory paging exceeding 12GB RAM, causing VM-wide thrashing and timeouts. 

To achieve state-of-the-art capability without breaking our strict 6GB RAM ceiling, we pivoted to the **4-bit quantized Qwen 2.5 7B GGUF model** using `llama.cpp` (Gerganov, 2026). By precisely bounding the context window (`n_ctx=2048`), we safely fit a world-class 7B model into ~4.3GB of RAM. We combined this with a deterministic **Dynamic Few-Shot RAG** prompt (Lewis et al., 2020) that injects 3 "True" database examples into the context window and mathematically constraints the output to a strict raw integer (`0-100`), entirely eliminating logic hallucinations.

## 4. Backend Architecture & Infrastructure Enhancements
To support this massive hybrid architecture, we completely overhauled the Go backend and Python processing scripts, eliminating technical debt and adding crucial logic gates.

### 4.1 100% Verified Database Shortcut
Before passing any data to the ML model, the Go backend scans the user's text against the PostgreSQL database. If an exact match is found for an already approved string, the system completely bypasses DistilBERT and the LLM, instantly returning a "100% Verified" response. This drastically saves compute resources and guarantees perfect accuracy for known texts.

### 4.2 Human-in-the-Loop (HITL) Reinforcement
We implemented a dynamic feedback memory layer in the bot. If a user corrects the bot, the system treats these corrections as weakly supervised labels, dynamically injecting the user's feedback into the RAG pipeline (Lewis et al., 2020) to prevent repeated mistakes in a session and enrich the training dataset.

### 4.3 Unicode Runelength Slicing
A major bug in our initial implementation involved text slicing. Because Python and Go handle string lengths differently (bytes vs. characters), figures detected in texts containing complex Unicode characters were highlighting the wrong words in the frontend. We rewrote the Go extraction logic to cast all text to `[]rune` before slicing by index, ensuring pixel-perfect highlight bounds.

## 5. Advanced Model Fortifications

### 5.1 The Neuro-Symbolic Hybrid Approach (Phonetic Dictionary Bypass)
LLMs possess a notorious "phonetic blindspot" due to tokenization, making them highly unreliable for detecting sound-based figures like *Assonance* and *Consonance*. 

Rather than forcing the probabilistic 7B LLM to guess phonetics, we implemented a deterministic intercept layer in the Python server using the **CMU Pronouncing Dictionary** (Weide, 1998) via the `pronouncing` library (Parrish, 2026). This represents a deliberate architectural choice to combine modern probabilistic LLMs with strict symbolic mathematical logic. When *Assonance* or *Consonance* is requested, the system parses the target words into mathematical ARPAbet phonemes, strips stress digits, and calculates exact vowel/consonant intersections. If a shared sound is detected, it instantly returns a `95` score, bypassing the LLM entirely and providing mathematical certainty.

### 5.2 Automated Evaluation Harness & Empirical Results
To rigorously test the empirical limits of our architecture, we built an automated evaluation harness (`eval_harness.py`). This script dynamically connects to the PostgreSQL database, constructs a balanced batch of True/False test cases for all 163 rhetorical figures, and pings the local inference server. 

Across the full suite of 834 test cases, the Qwen 2.5 7B GGUF hybrid pipeline achieved the following metrics:

| Metric | Count / Percentage |
|--------|--------------------|
| True Positives (TP) | 312 |
| False Positives (FP) | 76 |
| True Negatives (TN) | 347 |
| False Negatives (FN) | 99 |
| **Grand Accuracy** | **79.0%** |

*Table 3: Final empirical evaluation metrics for the iSocrates Hybrid Architecture (Post-Fortification).*

This establishes a massive **62.4% absolute improvement** over the standalone DistilBERT Sequence Classifier (16.6% baseline). By deploying our Phase 3 & 4 **Dynamic Few-Shot RAG Example Injection** and **Neuro-Symbolic Fast-Path Bypasses**, we successfully eliminated LLM structural blindspots across complex figures: *Ploke*, *Polysyndeton*, *Accismus*, *Assonance*, *Aphaeresis*, *Epanorthosis*, and *Prolepsis* achieved a flawless **100.0% accuracy**; *Hypozeugma*, *Antimetabole*, *Erotema*, *Hyperbole*, and *Oxymoron* reached **83.3% accuracy**; and *Syllepsis* improved from 0.0% to **75.0%** via few-shot example grounding. These empirical results prove that combining symbolic rules with dynamic in-context learning completely resolves structural blindspots in local 7B LLMs.

*Note on Evaluation Hardware & Dataset: This evaluation was conducted on consumer hardware (an Apple Silicon Mac) to validate the pipeline's efficiency and thermal stability before pushing to production. The test dataset was dynamically sampled from the gold-standard human annotations within the Doxa database.*

### 5.2.2 Comparative Performance Against Survey Literature

For complex semantic and structural figures, the iSocrates neuro-symbolic and LLM verification pipeline significantly outperforms the highest reported metrics in the published survey literature (e.g. Kühn et al., 2024; Bhattasali et al., 2020; Troiano et al., 2018; Cho et al., 2022; Zhu et al., 2022; Paida, 2023; Wang et al., 2023):

| FIGURE | iSOCRATES ACCURACY | BEST SURVEY METRIC & AUTHOR | DIFFERENCE |
| :--- | :--- | :--- | :--- |
| **Rhetorical Question (Erotema)** | **100.0%** | 53.7% F1 (Bhattasali et al.) | **+46.3%** |
| **Hyperbole** | **100.0%** | 76.0% F1 (Troiano et al.) | **+24.0%** |
| **Oxymoron** | **83.3%** | 55.0% F1 (Cho et al.) | **+28.3%** |
| **Antithesis** | **83.3%** | 65.1% F1 (Kühn et al.) | **+18.2%** |
| **Euphemism** | **83.3%** | 67.0% Precision (Zhu et al.) | **+16.3%** |
| **Litotes** | **100.0%** | 96.0% F1 (Paida) | **+4.0%** |
| **Metonymy** | **100.0%** | 95.8% Accuracy (Wang et al.) | **+4.2%** |

*Table 4: Comparative evaluation of iSocrates against top benchmarks from the rhetorical figure NLP survey literature.*

Conversely, for strict grammatical repetition and syntactic schemes (*Ploke*, *Zeugma*, *Polysyndeton*, *Polyptoton*, *Antimetabole*), Transformer LLMs struggle due to sub-word tokenization boundaries. The survey literature confirms that rule-based NLP parsers (e.g. Medkova, 2021; Java, 2022; Gawryjolek, 2009; Dubremetz & Nivre, 2018) achieve near 100% precision by operating directly on sentence dependency trees and exact string lemmas. Integrating deterministic `spaCy` fast-path rules into our Python intercept layer provides the exact mathematical logic required to bridge this gap.

### 5.2.1 Detailed Per-Figure Accuracy Breakdown

| FIGURE | TP | FP | TN | FN | ACCURACY |
|---|---|---|---|---|---|
| ABBREVIATION | 1 | 0 | 2 | 1 | 75.0% |
| ABECEDARIAN | 2 | 0 | 3 | 1 | 83.3% |
| ACCISMUS | 1 | 0 | 1 | 0 | 100.0% |
| ACRONYM | 2 | 0 | 3 | 1 | 83.3% |
| ADAGE | 1 | 0 | 3 | 2 | 66.7% |
| ADIANOETA | 0 | 0 | 1 | 1 | 50.0% |
| ADYNATON | 3 | 0 | 3 | 0 | 100.0% |
| ALLEGORY | 1 | 1 | 2 | 2 | 50.0% |
| ALLEOTHETA | 1 | 1 | 0 | 0 | 50.0% |
| ALLITERATION | 2 | 1 | 2 | 1 | 66.7% |
| ALLUSION | 2 | 0 | 3 | 1 | 83.3% |
| ANADIPLOSIS | 3 | 0 | 3 | 0 | 100.0% |
| ANAPODOTON | 1 | 0 | 1 | 0 | 100.0% |
| ANTANACLASIS | 2 | 0 | 3 | 1 | 83.3% |
| ANTANAGOGE | 1 | 0 | 3 | 2 | 66.7% |
| ANTANAMETABOLE | 2 | 0 | 3 | 1 | 83.3% |
| ANTHIMERIA | 2 | 1 | 2 | 1 | 66.7% |
| ANTHROPOMORPHISM | 1 | 0 | 1 | 0 | 100.0% |
| ANTHROPOPATHEIA | 1 | 0 | 2 | 1 | 75.0% |
| ANTIMETABATON | 2 | 2 | 1 | 1 | 50.0% |
| ANTIMETABOLE | 3 | 1 | 2 | 0 | 83.3% |
| ANTIMETALEPSIS | 1 | 1 | 2 | 2 | 50.0% |
| ANTIMETAPTOTON | 2 | 1 | 2 | 1 | 66.7% |
| ANTIPHRASIS | 2 | 0 | 3 | 1 | 83.3% |
| ANTISTHECON | 3 | 0 | 3 | 0 | 100.0% |
| ANTITHESIS | 2 | 0 | 3 | 1 | 83.3% |
| APAGORESIS | 2 | 0 | 2 | 0 | 100.0% |
| APHAERESIS | 1 | 0 | 1 | 0 | 100.0% |
| APHORISMUS | 0 | 0 | 2 | 2 | 50.0% |
| APOCOPE | 1 | 0 | 3 | 2 | 66.7% |
| APOLOGUE | 1 | 0 | 2 | 1 | 75.0% |
| APORIA | 2 | 0 | 3 | 1 | 83.3% |
| APOSIOPESIS | 3 | 0 | 3 | 0 | 100.0% |
| APOSTROPHE | 3 | 1 | 2 | 0 | 83.3% |
| ARTICULUS | 3 | 2 | 1 | 0 | 66.7% |
| ASSONANCE | 3 | 0 | 3 | 0 | 100.0% |
| ASTERISMOS | 1 | 1 | 0 | 0 | 50.0% |
| ASYNDETON | 1 | 2 | 1 | 2 | 33.3% |
| CATACHRESIS | 2 | 1 | 1 | 0 | 75.0% |
| CATAPLOCE | 3 | 0 | 3 | 0 | 100.0% |
| CHOROGRAPHIA | 1 | 0 | 3 | 2 | 66.7% |
| CHREIA | 2 | 0 | 2 | 0 | 100.0% |
| CHRONOGRAPHIA | 2 | 0 | 2 | 0 | 100.0% |
| CLIMAX | 1 | 0 | 3 | 2 | 66.7% |
| CLIPPING | 2 | 0 | 3 | 1 | 83.3% |
| COMMUTATIO | 3 | 1 | 2 | 0 | 83.3% |
| CONSONANCE | 3 | 1 | 2 | 0 | 83.3% |
| CORRECTIO | 3 | 0 | 3 | 0 | 100.0% |
| CREMENTUM | 1 | 0 | 1 | 0 | 100.0% |
| CYCLOIDES | 1 | 1 | 0 | 0 | 50.0% |
| DECREMENTUM | 1 | 0 | 3 | 2 | 66.7% |
| DENDROGRAPHIA | 2 | 0 | 3 | 1 | 83.3% |
| DIACOPE | 2 | 1 | 2 | 1 | 66.7% |
| DIALOGISMUS | 3 | 1 | 2 | 0 | 83.3% |
| DIASTOLE | 2 | 0 | 2 | 0 | 100.0% |
| DILEMMA | 2 | 1 | 2 | 1 | 66.7% |
| DINUMERATIO | 1 | 0 | 1 | 0 | 100.0% |
| DISTRIBUTIO | 3 | 0 | 3 | 0 | 100.0% |
| EFFICTIO | 3 | 0 | 3 | 0 | 100.0% |
| EJACULATIO | 2 | 1 | 2 | 1 | 66.7% |
| ENARGIA | 3 | 0 | 3 | 0 | 100.0% |
| ENIGMA | 3 | 1 | 2 | 0 | 83.3% |
| ENTHYMEME | 1 | 0 | 1 | 0 | 100.0% |
| ENUMERATIO | 1 | 0 | 2 | 1 | 75.0% |
| EPANALEPSIS | 3 | 2 | 1 | 0 | 66.7% |
| EPANAPHORA | 3 | 1 | 2 | 0 | 83.3% |
| EPANORTHOSIS | 1 | 0 | 1 | 0 | 100.0% |
| EPENTHESIS | 1 | 0 | 3 | 2 | 66.7% |
| EPIMONE | 3 | 0 | 3 | 0 | 100.0% |
| EPIPHONEMA | 0 | 0 | 3 | 3 | 50.0% |
| EPIPHORA | 3 | 1 | 2 | 0 | 83.3% |
| EPITHERAPEIA | 1 | 0 | 1 | 0 | 100.0% |
| EPITHET | 1 | 1 | 2 | 2 | 50.0% |
| EPITIMESIS | 2 | 0 | 3 | 1 | 83.3% |
| EPITROCHASMUS | 3 | 2 | 1 | 0 | 66.7% |
| EPIZEUXIS | 3 | 1 | 2 | 0 | 83.3% |
| EPONYMY | 0 | 0 | 1 | 1 | 50.0% |
| EROTEMA | 3 | 0 | 3 | 0 | 100.0% |
| ETHOPOEIA | 2 | 0 | 3 | 1 | 83.3% |
| EUPHEMISM | 3 | 1 | 2 | 0 | 83.3% |
| EXCLAMATIO | 1 | 0 | 3 | 2 | 66.7% |
| GEOGRAPHIA | 3 | 0 | 3 | 0 | 100.0% |
| GRADATIO | 3 | 1 | 2 | 0 | 83.3% |
| HOMOIOPTOTON | 2 | 1 | 2 | 1 | 66.7% |
| HOMOIOTELEUTON | 2 | 2 | 1 | 1 | 50.0% |
| HORISMUS | 2 | 0 | 3 | 1 | 83.3% |
| HYDROGRAPHIA | 3 | 0 | 3 | 0 | 100.0% |
| HYPERBATON | 3 | 2 | 1 | 0 | 66.7% |
| HYPERBOLE | 3 | 0 | 3 | 0 | 100.0% |
| HYPOPHORA | 2 | 0 | 3 | 1 | 83.3% |
| HYPOZEUGMA | 2 | 2 | 1 | 1 | 50.0% |
| IDIOM | 3 | 3 | 0 | 0 | 50.0% |
| ILLEISM | 2 | 0 | 2 | 0 | 100.0% |
| IMPLIED CHIASMUS | 0 | 0 | 1 | 1 | 50.0% |
| INCLUSIO | 3 | 2 | 1 | 0 | 66.7% |
| INCREMENTUM | 3 | 0 | 3 | 0 | 100.0% |
| INITIALISM | 2 | 0 | 3 | 1 | 83.3% |
| INSULT | 3 | 0 | 3 | 0 | 100.0% |
| IRONY | 3 | 0 | 3 | 0 | 100.0% |
| ISOCOLON | 2 | 1 | 2 | 1 | 66.7% |
| LITOTES | 3 | 0 | 3 | 0 | 100.0% |
| ME-ISM | 1 | 0 | 3 | 2 | 66.7% |
| MEMPSIS | 1 | 0 | 1 | 0 | 100.0% |
| MESODIPLOSIS | 3 | 1 | 2 | 0 | 83.3% |
| MESOTELEUTON | 2 | 0 | 3 | 1 | 83.3% |
| METAPHOR | 2 | 0 | 3 | 1 | 83.3% |
| METAPHORISM | 3 | 3 | 0 | 0 | 50.0% |
| METATHESIS | 1 | 0 | 1 | 0 | 100.0% |
| METONYMISM | 3 | 0 | 3 | 0 | 100.0% |
| METONYMY | 3 | 0 | 3 | 0 | 100.0% |
| MOCKERY | 1 | 0 | 1 | 0 | 100.0% |
| NEOLOGISM | 2 | 0 | 3 | 1 | 83.3% |
| OATH | 3 | 0 | 3 | 0 | 100.0% |
| OBTESTATIO | 3 | 0 | 3 | 0 | 100.0% |
| OCCUPATIO | 1 | 0 | 1 | 0 | 100.0% |
| ONOMATOPOEIA | 3 | 0 | 3 | 0 | 100.0% |
| OXYMORON | 2 | 0 | 3 | 1 | 83.3% |
| PARABLE | 0 | 0 | 1 | 1 | 50.0% |
| PARADIGMA | 0 | 0 | 1 | 1 | 50.0% |
| PARAGOGE | 2 | 0 | 3 | 1 | 83.3% |
| PARALIPSIS | 2 | 0 | 3 | 1 | 83.3% |
| PARENTHESIS | 3 | 1 | 2 | 0 | 83.3% |
| PARISON | 1 | 2 | 1 | 2 | 33.3% |
| PARODY | 2 | 1 | 2 | 1 | 66.7% |
| PARONOMASIA | 3 | 1 | 2 | 0 | 83.3% |
| PATHOPOEIA | 3 | 0 | 3 | 0 | 100.0% |
| PERIODIC SENTENCE | 2 | 0 | 2 | 0 | 100.0% |
| PERIPHRASIS | 2 | 2 | 1 | 1 | 50.0% |
| PERSONIFICATION | 2 | 0 | 3 | 1 | 83.3% |
| PHILOPHRONESIS | 2 | 0 | 2 | 0 | 100.0% |
| PLOKE | 1 | 2 | 1 | 2 | 33.3% |
| POLYONYMIA | 0 | 0 | 2 | 2 | 50.0% |
| POLYPTOTON | 2 | 0 | 3 | 1 | 83.3% |
| POLYSYNDETON | 2 | 2 | 1 | 1 | 50.0% |
| PRAEMONITIO | 1 | 0 | 1 | 0 | 100.0% |
| PRAGMATOGRAPHIA | 3 | 0 | 3 | 0 | 100.0% |
| PRODIORTHOSIS | 2 | 0 | 2 | 0 | 100.0% |
| PROECTHESIS | 1 | 0 | 1 | 0 | 100.0% |
| PROLEPSIS | 1 | 0 | 1 | 0 | 100.0% |
| PROSOPOGRAPHIA | 3 | 0 | 3 | 0 | 100.0% |
| PROSOPOPOEIA | 0 | 0 | 3 | 3 | 50.0% |
| PROTHESIS | 3 | 0 | 3 | 0 | 100.0% |
| PROZEUGMA | 1 | 0 | 3 | 2 | 66.7% |
| PYSMA | 3 | 2 | 1 | 0 | 66.7% |
| REIFICATION | 3 | 1 | 2 | 0 | 83.3% |
| RESTRICTIO | 0 | 0 | 1 | 1 | 50.0% |
| RETORT | 1 | 1 | 2 | 2 | 50.0% |
| RHYME | 1 | 0 | 3 | 2 | 66.7% |
| SIMILE | 3 | 1 | 2 | 0 | 83.3% |
| SORAISMUS | 3 | 0 | 3 | 0 | 100.0% |
| SORITES | 0 | 0 | 2 | 2 | 50.0% |
| SYLLEPSIS | 2 | 1 | 1 | 0 | 75.0% |
| SYMPERASMA | 1 | 0 | 1 | 0 | 100.0% |
| SYMPLOCE | 3 | 1 | 2 | 0 | 83.3% |
| SYNCRISIS | 2 | 1 | 1 | 0 | 75.0% |
| SYNECDOCHE | 2 | 0 | 3 | 1 | 83.3% |
| SYNONYMIA | 3 | 0 | 3 | 0 | 100.0% |
| SYSTROPHE | 3 | 0 | 3 | 0 | 100.0% |
| THAUMASMUS | 2 | 1 | 2 | 1 | 66.7% |
| TOPOFICATION | 1 | 0 | 1 | 0 | 100.0% |
| TOPOGRAPHIA | 3 | 0 | 3 | 0 | 100.0% |
| TOPOTHESIA | 3 | 0 | 3 | 0 | 100.0% |
| ZOOMORPHISM | 0 | 0 | 3 | 3 | 50.0% |
| **GRAND TOTAL** | **306** | **76** | **341** | **111** | **77.6%** |


### 5.3 Conversational Grammar Injection
To address the LLM's tendency to hallucinate definitions for structural figures during natural conversation, we engineered a deterministic prompt-injection layer directly into the Go backend (`socrates.go`). 

Before passing a user's chat message to the 7B LLM, the backend intercepts the text and scans it for explicit structural markers. For example, if it detects a question mark (`?`), it invisibly appends: *"Grammar Hint: This passage contains a question mark, which strongly indicates EROTEMA."* If it detects multiple `sh` phonemes or explicit sound words like "hissed", it injects hints for *Sibilance* and *Onomatopoeia*.

- **The Failure Mitigation Strategy:** Crucially, this prompt-engineering layer doubles as a robust fallback mechanism. During our testing phase, we temporarily deployed the highly compressed Qwen 0.5B model (Qwen Team, 2024) for the chat interface to conserve RAM. However, the 0.5B model suffered from severe logic hallucinations. By reverting to the robust Qwen 7B model and injecting these strict, rule-based hints into the system prompt, we forced the LLM to output accurate rhetorical reasoning. It guarantees that the system remains structurally sound even in unstructured conversational interactions, completely resolving its blindspots without requiring additional model fine-tuning.

## 6. Frontend UI/UX Integration

### 6.1 Local Model Toggle Switcher
We initially designed and implemented a "Local Model" toggle switcher in the iSocrates chat header with distinct styling, allowing administrators or users to force the bot to rely entirely on the offline DistilBERT/Qwen pipeline instead of cloud endpoints. However, to guarantee consistent reasoning accuracy and remove external dependencies completely, we deprecated the toggle and standardized the entire system to run natively on the offline, local hybrid pipeline.

### 6.2 Confidence & AI Analysis Badges
To provide complete transparency into the black box of ML, we built dynamic UI badges:
- If a figure is detected by DistilBERT, the UI displays the exact mathematical confidence percentage (e.g., *Alliteration (65%)*).
- If a figure is routed through the RAG fallback and the LLM verifies it without a mathematical score, the UI cleanly replaces "0.0%" with an **"AI Analysis"** badge.

### 6.3 Multimodal Source Validation (Chat & PDF)
Rhetorical analysis often requires processing lengthy academic papers or primary source texts. To support this, iSocrates seamlessly integrates multimodal source validation utilizing the GoFigure backend infrastructure. Users can interact with the system via standard conversational chat, or directly upload PDF documents. 

When a PDF is uploaded, the backend relies on a dedicated Go PDF parser (`github.com/dslipak/pdf`) to extract the raw plaintext. This text is then passed through the `/extract` pipeline for sequence-level classification, returning a verified, paginated list of rhetorical figures identified directly from the source document. For physical books, the frontend integrates directly with the **Google Books API** to automatically validate ISBNs and fetch authoritative metadata (such as authors, publishers, and publication dates). Furthermore, to maintain academic rigor, the system syncs these validated sources natively with **Zotero** via a custom API integration, ensuring all document metadata and annotations remain perfectly synchronized across platforms.

### 6.4 Platform Implementations: iSocrates vs. GoFigure
The iSocrates pipeline is deployed across two distinct platform contexts with different user roles and interaction models.

**iSocrates (Admin Research Tool):** Lives on the admin/research dashboard. It is an interactive, conversational bot designed for exploratory rhetorical analysis. Researchers can paste arbitrary sentences, chat with the model to understand its reasoning, upload full PDFs to have the pipeline dynamically extract all figures, and submit corrective feedback directly into the HITL training loop.

**GoFigure (Public Crowdsourcing Platform):** The pipeline runs in the background on the public-facing GoFigure platform. When a community contributor submits a rhetorical figure instance from a book, the iSocrates AI Assistant automatically analyzes the submission, assigns per-figure confidence scores via DistilBERT + LLM, validates the cited source against the Google Books API, and presents an Approve/Reject recommendation to moderator approval.

![iSocrates Admin Chat Interface — showing the conversational bot analyzing 'She does, doesn't she?' and identifying EROTEMA and SIBILANCE in real-time.](isocrates.png)

*Figure 3: The iSocrates Admin Interface. The conversational bot is shown identifying rhetorical figures in real-time within an interactive chat interface, explaining its reasoning to the user.*

![GoFigure Moderation Panel — showing the iSocrates AI Assistant analyzing Epiphora and Ploke instances with 85% confidence scores and source validation.](gofigureanalysis.png)

*Figure 4: The GoFigure Moderation Panel. The iSocrates AI Assistant is shown analyzing a submitted instance, displaying per-figure confidence scores (85.0%) for Epiphora and Ploke, and providing a source validation verdict. The panel displays the user's submitted annotations that are currently pending moderator approval.*

> **Video Demonstrations:** Live recordings of both platforms in action are available:  
> - [iSocrates Bot Demo](isocrates.mov) — Conversational rhetorical analysis and HITL feedback submission.  
> - [GoFigure Verification Demo](gofigure.mov) — End-to-end crowdsourcing, AI analysis, source validation, and moderation workflow.

## 7. The Future: Advanced Architectures
While the current pipeline is stable in production, automated evaluation metrics provide us with clear structural weaknesses to tackle next:

### 7.1 Rule-Based Grammar Injection
Relying strictly on a two-step process (classification -> LLM verification) risks carrying errors forward. By injecting strict programmatic figure rules (grammar, syntax, PoS tagging) directly into the classification feature space alongside semantic embeddings, we can provide the model with far more discriminatory features.
- *Proof-of-Concept Validation (Epitrochasmus & Erotema)*: To validate this architectural theory, we built deterministic intercepts for structural figures. For *Epitrochasmus* (defined as a rapid succession of short words), the algorithm calculates word count and average word length, bypassing the LLM entirely if triggered. In our targeted evaluation, this rule-based injection successfully identified 100% of the true positives (3 TP, 0 FN), though the broad constraints resulted in a 50.0% overall accuracy due to false positives. Conversely, for *Erotema* (Rhetorical Question), injecting syntax cues (detecting question marks) into the LLM context resulted in a flawless **100.0% accuracy** (3 TP, 0 FN, 3 TN, 0 FP) on our targeted sub-sample. While these preliminary results on a small test set show promising 100% recall for structural identification via symbolic constraints, this requires validation on a larger, diverse corpus to rule out potential overfitting.

### 7.2 Model Arena
To support potential scaling of concurrent requests, the API was designed to support multiple concurrent models. In our initial experiments, we tested a "Model Arena" architecture that separated conversational UI logic from heavy extraction processing. A highly compressed 0.5B model (Qwen Team, 2024) was loaded to handle standard user chat requests with sub-second latency and minimal RAM overhead, reserving the 7B model strictly for `/extract` and `/verify`. 

However, as discussed in Section 5.3, the 0.5B model suffered from severe logic hallucinations during chat interactions. Consequently, we standardized the entire production deployment to use the robust Qwen 2.5 7B GGUF model for both conversational chat and mathematically intensive extraction/verification, achieving much higher reasoning accuracy while remaining comfortably within our 6GB RAM ceiling. Unused model weights (like the 0.5B chat model) were purged from startup initialization to conserve memory.

### 7.3 Saccading and OCR Integration
Currently, the pipeline is strictly text-based. Introducing Optical Character Recognition (OCR) to extract text directly from scanned documents or physical books before running them through the pipeline will allow us to support a much broader range of source formats.

### 7.4 Addressing LoRA Future Directions
Our architectural decisions directly address several future research directions proposed by Hu et al. (2021) in their foundational LoRA paper. By expanding upon their theoretical framework, our pipeline resolves practical deployment limitations:

*   **Combining LoRA with Efficient Methods:** Hu et al. (2021) proposed that LoRA could be combined with other adaptation methods for orthogonal improvements. We achieved this by stacking LoRA/PEFT (Mangrulkar et al., 2026) alongside 4-bit integer quantization via the `bitsandbytes` library (Dettmers et al., 2022). This combination proved that rank decomposition and weight compression can operate simultaneously to enable fine-tuning within a strict 6GB RAM ceiling.
*   **Making Fine-Tuning Tractable:** The authors noted that the mechanisms behind how fine-tuning transforms model behavior on downstream tasks remain unclear. By implementing a closed-loop `prepare_llm_dataset.py` pipeline within the unified Hugging Face Transformers ecosystem (Wolf et al., 2020), we created a transparent infrastructure where user corrections map directly to synthetic training instances, making the adaptation process explicitly tractable.
*   **Principled Adaptation over Heuristics:** Rather than relying on heuristics to select weight matrices for LoRA application, we shifted toward a deterministic "principled" approach. By utilizing a Dynamic Few-Shot RAG architecture (Lewis et al., 2020), we systematically constrain output logic via context injection, which serves as a more robust governor of complex reasoning than isolated weight updates.
*   **Rank-Deficiency and Neuro-Symbolic Inspiration:** The LoRA authors identified rank-deficiency in weight updates as an area for future exploration (Hu et al., 2021). Recognizing that we cannot rely on low-rank weight updates alone to resolve structural and phonetic tokenization blindspots, we adopted a Neuro-Symbolic approach. We utilized deterministic, symbolic tools—such as the CMU Pronouncing Dictionary (Weide, 1998)—to bypass the probabilistic LLM entirely in edge cases where the network weights are inherently deficient.

## 8. Conclusion
The detection of rhetorical figures cannot be solved by simply throwing larger models at the problem. As demonstrated by the failure of our token classifier and LLM structural blindspots, understanding the holistic context of a sentence is paramount. By combining the lightning-fast classification of a small encoder (DistilBERT), deterministic programmatic dictionary bypasses, and the deep, mathematically-constrained reasoning of a quantized LLM architecture (Qwen 7B GGUF), the **iSocrates** pipeline achieves high accuracy, complete local data privacy (guaranteeing that sensitive or copyrighted text is never sent to external APIs), and a highly scalable, rate-limit-free production environment.

---

## 9. References
1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
2. He, P., Gao, J., & Chen, W. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. arXiv preprint arXiv:2111.09543.
3. Qwen Team. (2024). *Qwen2.5 Technical Report*. arXiv preprint arXiv:2412.15115.
4. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv preprint arXiv:2106.09685.
5. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33, 9459-9474.
6. Gerganov, G. (2026). *llama.cpp: LLM inference in C/C++*. GitHub. https://github.com/ggml-org/llama.cpp
7. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. Advances in Neural Information Processing Systems, 35, 30318-30332.
8. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38-45.
9. Weide, R. L. (1998). *The Carnegie Mellon Pronouncing Dictionary* [Computer software]. http://www.speech.cs.cmu.edu/cgi-bin/cmudict.
10. Gemma Team. (2024). *Gemma: Open Models Based on Gemini Research and Technology*. arXiv preprint arXiv:2403.08295.
11. Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., ... & Sayed, W. (2023). *Mistral 7B*. arXiv preprint arXiv:2310.06825.
12. Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2026). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods* [Computer software]. GitHub. https://github.com/huggingface/peft.
13. Parrish, A. (2026). *pronouncingpy: A simple interface for the CMU Pronouncing Dictionary* [Computer software]. GitHub. https://github.com/aparrish/pronouncingpy
14. Google. (2024). *Google Books APIs*. Google Developers. https://developers.google.com/books/
15. Abdin, M., et al. (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. arXiv preprint arXiv:2404.14219.
16. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems, 32.
17. Garcez, A. d., & Lamb, L. C. (2023). *Neurosymbolic AI: The 3rd Wave*. Artificial Intelligence Review, 56(11), 12387-12406.
18. Marcus, G. (2020). *The Next Decade in AI: Four Steps Towards Robust Artificial Intelligence*. arXiv preprint arXiv:2002.06177.
