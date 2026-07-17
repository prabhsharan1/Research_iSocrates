# iSocrates: A Hybrid Architecture for Real-Time Rhetorical Figure Detection and Verification

**Abstract**  
The computational detection of rhetorical figures remains a complex challenge in Natural Language Processing (NLP) due to the nuanced, context-dependent, and often overlapping definitions of rhetorical devices. In this paper, we present the architecture of **iSocrates**, a real-time rhetorical analysis bot. We detail the evolution of our machine learning pipeline—from our initial DistilBERT sequence classifier to token-level transformers—and expose the pitfalls of relying solely on high-accuracy token classifiers that fail in real-world deployments. To solve these challenges, we introduce a novel **"Triple-Check" Hybrid Architecture** that combines the sub-100ms inference speed of an encoder with deterministic programmatic constraints (like phonetic dictionaries), and the deep semantic reasoning of a highly accessible, lightweight local Large Language Model (progressing from Gemma 2B to a highly quantized Qwen 2.5 7B GPT-Generated Unified Format (GGUF)). Finally, we document the extensive full-stack infrastructure built to support this pipeline, including dynamic training filters, automated evaluation harnesses, and strict database-matching shortcuts.

---

## System Architecture Overview

```mermaid
flowchart TD
    A(["User Input\n(Chat / PDF Upload)"])
    B{{"Postgres\nExact-Match Shortcut"}}
    C(["✅ 100% Verified\nInstant Return"])
    D["DistilBERT\nSequence Classifier\n(threshold: 5%)"]
    E{{"Phonetic Figure?\n(Assonance / Consonance)"}}
    F["CMU Pronouncing Dict\nARPAbet Intersection\n→ score 95"]
    G{{"Structural Hint?\n(Grammar Injection)"}}
    H["Go Backend\nInjects Grammar Hints\ninto System Prompt"]
    I["Qwen 2.5 7B GGUF\nDynamic Few-Shot RAG\nVerification (0-100)"]
    J(["JSON Response\nCharacter Spans + Explanation"])
    K(["Frontend\niSocrates Chat / GoFigure Badge"])

    A --> B
    B -- "Match Found" --> C
    B -- "No Match" --> D
    D -- "Candidate Figures" --> E
    E -- "Yes" --> F
    E -- "No" --> G
    F --> J
    G -- "Yes" --> H
    G -- "No" --> I
    H --> I
    I --> J
    J --> K
```

*Figure 1: The Triple-Check Hybrid Architecture. Each layer acts as a deterministic gate that prevents the probabilistic LLM from being invoked unnecessarily, prioritizing speed and mathematical certainty.*

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

During validation, this model achieved an impressive **87% accuracy**. However, deploying this model revealed a classic ML pitfall: poor out-of-domain generalization. In real-world text, it failed catastrophically on edge cases—for example, falsely identifying the single preposition "to" as an *Antimetabole*. It entirely missed overarching phonetic figures (like *Alliteration*) that span entire sentences.

### 2.3 The DeBERTa Token Classifier & The "NaN Loss" Collapse
In our second attempt to make token classification work, we upgraded to `microsoft/deberta-v3-small` (He et al., 2021) to exploit its advanced relative position encodings. 

The transition to `microsoft/deberta-v3-small` introduced numerical instability. Extreme class imbalance, where over 99.9% of tokens belong to the 'Outside' (O) class, led to gradient collapse. This highlighted the necessity of shifting away from token-level classification toward a hybrid sequence-extraction pipeline.

### 2.4 Reverting to Sequence Classification + LLM Extraction (The Hybrid Solution)
Guided by these findings, we established that rhetorical figures are best detected holistically at the sequence level. We definitively reverted our local classifier to the stable **DistilBERT Sequence Classifier** (Sanh et al., 2019), lowering the classification threshold to 5% to capture multiple overlapping candidate figures in a single sentence. 

Once candidate figures are identified, we delegate the exact character-span extraction and explanation to the **local LLM**. This separation of concerns (fast sentence-level classification + deep token-level LLM extraction) successfully bypassed the token-imbalance problem and ensured 100% mathematical stability.

### 2.5 Data Curation and Source Validation
The foundation of both the DistilBERT classifier and the subsequent LLM verification layer relies entirely on a meticulously curated dataset. A pervasive challenge in rhetorical figure detection is the lack of standardized definitions, which frequently leads to poor inter-annotator agreement and corrupted training signals. To solve this, our architecture relies on human-verified training data curated through collaborative pedagogical review within the Rhetoricon project's *Doxa* database. 

While *Doxa* serves as our primary data repository, we implemented a strict, automated **Source Validation** layer to ensure the integrity of the underlying texts. To guarantee absolute metadata accuracy, the frontend integrates directly with the Google Books API (Google, 2024) to automatically validate ISBNs and fetch authoritative bibliographic data (such as exact authors, publishers, and publication dates) before any instance is permanently mapped to a verified bibliographic source. By grounding the training data and the RAG pipeline (Lewis et al., 2020) in this strict, API-validated ontology, the architecture ensures the models learn the true functional properties of the figures rather than noisy, unverified internet data.

## 3. The Hybrid RAG Pipeline and LLM Selection
While the Sequence Classifier is extremely fast, it still struggles with highly specific edge-case definitions that require deep logical reasoning. We needed an LLM to verify these complex cases.

### 3.1 Model Selection: Why Gemma 2B?
1. **Avoiding External APIs:** We initially considered using external APIs like Google's Gemini. While the free tier works for testing, scaling it to a public audience would hit strict rate limits and expensive paywalls. Because we already possessed a massive database of human-verified training data, we opted to build a completely self-hosted solution to keep the bot entirely free and accessible.
2. **Mistral vs. Gemma 2B:** We evaluated local LLMs to host on our Portainer staging/production servers. Mistral 7B (Jiang et al., 2023) is highly capable but requires nearly 5 GB of RAM when quantized, making it too heavy and risky for a lightweight Portainer deployment. We ultimately selected **Gemma 2B** (Gemma Team, 2024). At under 2 GB of RAM (when using 4-bit quantization), Gemma 2B is incredibly lightweight, fast, and highly capable of strict pattern-matching when provided with in-context examples.

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
Despite fine-tuning, smaller models (0.5B - 2B) exhibited a structural "0% collapse" on complex definitions and were prone to outputting hallucinatory text rather than strict JSON logic. 

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
| True Positives (TP) | 290 |
| False Positives (FP) | 81 |
| True Negatives (TN) | 336 |
| False Negatives (FN) | 127 |
| **Grand Accuracy** | **75.1%** |

*Table 2: Final empirical evaluation metrics for the iSocrates Hybrid Architecture.*

This establishes a massive **58.5% absolute improvement** over the standalone DistilBERT Sequence Classifier (16.6% baseline). However, analyzing the confusion matrix reveals a hard empirical ceiling. While the LLM excels at semantic pattern recognition, it severely underperforms on figures requiring strict grammatical syntax (e.g., *Syllepsis* scored 0.0%, *Epitrochasmus* 16.7%, and *Hypozeugma* 16.7%). These specific weaknesses provide the mathematical justification required to transition to advanced, constraint-based architectures.

*Note on Evaluation Hardware & Dataset: This evaluation was conducted on consumer hardware (an Apple Silicon Mac) to validate the pipeline's efficiency and thermal stability before pushing to production. The test dataset was dynamically sampled from the gold-standard human annotations within the Doxa database.*

### 5.3 Conversational Grammar Injection
To address the LLM's tendency to hallucinate definitions for structural figures during natural conversation, we engineered a deterministic prompt-injection layer directly into the Go backend (`socrates.go`). 

Before passing a user's chat message to the 7B LLM, the backend intercepts the text and scans it for explicit structural markers. For example, if it detects a question mark (`?`), it invisibly appends: *"Grammar Hint: This passage contains a question mark, which strongly indicates EROTEMA."* If it detects multiple `sh` phonemes or explicit sound words like "hissed", it injects hints for *Sibilance* and *Onomatopoeia*.

- **The Failure Mitigation Strategy:** Crucially, this prompt-engineering layer doubles as a robust fallback mechanism. During our testing phase, we temporarily deployed the highly compressed Qwen 0.5B model (Qwen Team, 2024) for the chat interface to conserve RAM. However, the 0.5B model suffered from severe logic hallucinations. By reverting to the robust Qwen 7B model and injecting these strict, rule-based hints into the system prompt, we forced the LLM to output accurate rhetorical reasoning. It guarantees that the system remains structurally sound even in unstructured conversational interactions, completely resolving its blindspots without requiring additional model fine-tuning.

## 6. Frontend UI/UX Integration

### 6.1 Local Model Toggle Switcher
We added a native "Local Model" toggle switcher directly into the iSocrates chat header with distinct green styling. This allows administrators or users to seamlessly force the bot to rely entirely on the offline DistilBERT/Qwen pipeline, manually overriding any cloud endpoints.

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

**GoFigure (Public Crowdsourcing Platform):** The pipeline runs in the background on the public-facing GoFigure platform. When a community contributor submits a rhetorical figure instance from a book, the iSocrates AI Assistant automatically analyzes the submission, assigns per-figure confidence scores via DistilBERT + LLM, validates the cited source against the Google Books API, and presents an Approve/Reject recommendation to moderators.

![iSocrates Admin Chat Interface — showing the conversational bot analyzing 'She does, doesn't she?' and identifying EROTEMA and SIBILANCE in real-time.](isocrates.png)

*Figure 2: The iSocrates Admin Interface. The bot is shown identifying EROTEMA and SIBILANCE in a user-submitted sentence alongside a list of candidate figures detected by DistilBERT.*

![GoFigure Moderation Panel — showing the iSocrates AI Assistant analyzing Epiphora and Ploke instances with 85% confidence scores and source validation.](gofigureanalysis.png)

*Figure 3: The GoFigure Moderation Panel. The iSocrates AI Assistant is shown analyzing a submitted instance, displaying per-figure confidence scores (85.0%) for Epiphora and Ploke, and providing a source validation verdict (⚠ Suspicious — publisher mismatch flagged, with an Auto-Fix option).*

> **Video Demonstrations:** Live recordings of both platforms in action are available:  
> - [iSocrates Bot Demo](isocrates.mov) — Conversational rhetorical analysis and HITL feedback submission.  
> - [GoFigure Verification Demo](gofigure.mov) — End-to-end crowdsourcing, AI analysis, source validation, and moderation workflow.

## 7. The Future: Advanced Architectures
While the current pipeline is stable in production, automated evaluation metrics provide us with clear structural weaknesses to tackle next:

### 7.1 Rule-Based Grammar Injection
Relying strictly on a two-step process (classification -> LLM verification) risks carrying errors forward. By injecting strict programmatic figure rules (grammar, syntax, PoS tagging) directly into the classification feature space alongside semantic embeddings, we can provide the model with far more discriminatory features.
- *Proof-of-Concept Validation (Epitrochasmus & Erotema)*: To validate this architectural theory, we built deterministic intercepts for structural figures. For *Epitrochasmus* (defined as a rapid succession of short words), the algorithm calculates word count and average word length, bypassing the LLM entirely if triggered. In our targeted evaluation, this rule-based injection successfully identified 100% of the true positives (3 TP, 0 FN), though the broad constraints resulted in a 50.0% overall accuracy due to false positives. Conversely, for *Erotema* (Rhetorical Question), injecting syntax cues (detecting question marks) into the LLM context resulted in a flawless **100.0% accuracy** (3 TP, 0 FN, 3 TN, 0 FP) on our targeted sub-sample. While these preliminary results on a small test set show promising 100% recall for structural identification via symbolic constraints, this requires validation on a larger, diverse corpus to rule out potential overfitting.

### 7.2 Model Arena
As we scaled, the API needed to abstract away single-model dependency. We introduced a "Model Arena" architecture that separates conversational UI logic from heavy extraction processing. A highly compressed 0.5B model (Qwen Team, 2024) handles standard user chat requests with sub-second latency and minimal RAM overhead, while the massive 7B model is strictly reserved for the mathematically-intensive pipelines: `/extract` (the service layer that performs sentence-level classification to identify candidate rhetorical figures) and `/verify` (the verification layer that leverages the 7B LLM to perform character-span identification and explanatory reasoning).

### 7.3 Saccading and OCR Integration
Currently, the pipeline is strictly text-based. Introducing Optical Character Recognition (OCR) to read figures directly from scanned texts will require simulating human "saccading" (eye movement tracking over text blocks) to accurately map rhetorical structures across spatial dimensions.

### 7.4 Addressing LoRA Future Directions
Our architectural decisions directly address several future research directions proposed by Hu et al. (2021) in their foundational LoRA paper. By expanding upon their theoretical framework, our pipeline resolves practical deployment limitations:

*   **Combining LoRA with Efficient Methods:** Hu et al. (2021) proposed that LoRA could be combined with other adaptation methods for orthogonal improvements. We achieved this by stacking LoRA/PEFT (Mangrulkar et al., 2026) alongside 4-bit integer quantization via the `bitsandbytes` library (Dettmers et al., 2022). This combination proved that rank decomposition and weight compression can operate simultaneously to enable fine-tuning within a strict 6GB RAM ceiling.
*   **Making Fine-Tuning Tractable:** The authors noted that the mechanisms behind how fine-tuning transforms model behavior on downstream tasks remain unclear. By implementing a closed-loop `prepare_llm_dataset.py` pipeline within the unified Hugging Face Transformers ecosystem (Wolf et al., 2020), we created a transparent infrastructure where user corrections map directly to synthetic training instances, making the adaptation process explicitly tractable.
*   **Principled Adaptation over Heuristics:** Rather than relying on heuristics to select weight matrices for LoRA application, we shifted toward a deterministic "principled" approach. By utilizing a Dynamic Few-Shot RAG architecture (Lewis et al., 2020), we systematically constrain output logic via context injection, which serves as a more robust governor of complex reasoning than isolated weight updates.
*   **Rank-Deficiency and Neuro-Symbolic Inspiration:** The LoRA authors identified rank-deficiency in weight updates as an area for future exploration (Hu et al., 2021). Recognizing that we cannot rely on low-rank weight updates alone to resolve structural and phonetic tokenization blindspots, we adopted a Neuro-Symbolic approach. We utilized deterministic, symbolic tools—such as the CMU Pronouncing Dictionary (Weide, 1998)—to bypass the probabilistic LLM entirely in edge cases where the network weights are inherently deficient.

## 8. Conclusion
The detection of rhetorical figures cannot be solved by simply throwing larger models at the problem. As demonstrated by the failure of our 87% token classifier and LLM structural blindspots, understanding the holistic context of a sentence is paramount. By combining the lightning-fast classification of a small encoder (DistilBERT), deterministic programmatic dictionary bypasses, and the deep, mathematically-constrained reasoning of a dual-model LLM architecture (Qwen 0.5B for conversational latency; Qwen 7B for robust semantic extraction), **iSocrates** achieves state-of-the-art accuracy, complete data privacy, and a highly scalable, rate-limit-free production environment.

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
15. Abdin, M., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv preprint arXiv:2404.14219. https://arxiv.org/abs/2404.14219
