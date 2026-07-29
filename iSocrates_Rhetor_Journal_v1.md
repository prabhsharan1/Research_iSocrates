# iSocrates: A Neuro-Symbolic Hybrid System for Computational Detection of Rhetorical Figures

**Abstract**  
Computational approaches to rhetorical figure detection have historically focused on a narrow set of popular devices—metaphor, irony, sarcasm—while over 400 of the 433 figures catalogued in classical ontologies such as the *Silva Rhetoricae* (Burton, 2007) remain computationally unaddressed (Kühn et al., 2024a). This paper presents **iSocrates**, a hybrid detection system developed within the Rhetoricon project (Harris & Di Marco, 2025) that combines symbolic linguistic rules, statistical classification, and a locally-run language model to identify 163 rhetorical figures — including many lesser-studied figures of repetition, sound, and syntactic structure that resist purely statistical approaches. Evaluated against 642 gold-standard instances drawn from the Rhetoricon *Doxa* database, iSocrates achieves 80.5% accuracy, a substantial improvement over a single statistical classifier baseline (16.6%). We discuss what this architecture reveals about the specific linguistic properties — tokenization-level fragmentation, phonetic invisibility, boundary ambiguity — that make rhetorical figures resistant to purely statistical language models, and what a hybrid symbolic/statistical approach can and cannot recover.

---

## Abbreviations and Terms

| Abbreviation / Term | Full Form / Definition | Description in iSocrates Context |
|---|---|---|
| **HITL** | Human-in-the-Loop | Continuous feedback loop where moderator corrections are used to guide future system behavior. |
| **RAG** | Retrieval-Augmented Generation | Technique for supplying a language model with relevant reference examples at the moment of judgment, rather than relying solely on its pre-trained knowledge. |
| **NLP** | Natural Language Processing | The subfield of linguistics and computer science concerned with computational processing of human language. |
| **LLM** | Large Language Model | A generative statistical language model (here, a locally-run open-weight model) used for semantic figure verification. |
| **BIO Tagging** | Beginning, Inside, Outside Tagging | A standard scheme for marking which words belong to a figure's span within a sentence. |
| **PoS** | Part-of-Speech | Grammatical word category (noun, verb, adjective, etc.), used in syntactic detection rules. |
| **CMU Dict** | Carnegie Mellon University Pronouncing Dictionary | A reference dictionary mapping English words to their phonetic sound sequences, used to detect sound-based figures independent of spelling. |
| **ARPAbet** | Advanced Research Projects Agency Phonetic Alphabet | A standardized code for representing English speech sounds, underlying the phonetic detection method described in Section 4. |
| **TP / FP / TN / FN** | True Positive / False Positive / True Negative / False Negative | The four outcome categories used to compute the accuracy, precision, recall, and F1 statistics reported in Section 5. |
| **DOI** | Digital Object Identifier | A persistent identifier for a specific electronic publication, distinct from a URL, which locates rather than identifies a resource. |

*(A fuller technical glossary, including infrastructure- and implementation-specific terms, is provided in the Implementation Notes appendix.)*

---

## 1. Introduction

Rhetorical figures — *Alliteration*, *Antimetabole*, *Mesodiplosis*, *Oxymoron*, and hundreds of others catalogued since antiquity — are constitutive of persuasive, literary, and everyday language (Fahnestock, 2002). Yet as Kühn et al. (2024a) demonstrate in their survey of the field, computational NLP has almost entirely neglected this diversity: over 90% of existing detection literature addresses only a handful of well-known tropes, leaving the great majority of the 433 figures in the *Silva Rhetoricae* (Burton, 2007) essentially untouched by computational methods.

This neglect is not accidental. As we argue in this paper, most rhetorical figures resist the statistical language models that have otherwise transformed NLP, for reasons rooted in how these figures actually function linguistically:

1. **Fragmentation at the token level.** Modern language models process text by breaking words into arbitrary subword fragments (e.g., splitting *"alliteration"* into `all`, `it`, `era`, `tion`). This fragmentation severs exactly the whole-word and cross-word relationships — repetition, inflectional variation, phonetic recurrence — that many rhetorical figures depend on.
2. **Phonetic invisibility.** Figures like *Alliteration*, *Assonance*, and *Consonance* are defined by how words *sound*, not how they are spelled. A model trained purely on written text has no direct access to this information.
3. **Boundary ambiguity and definitional inconsistency.** As Kühn et al. (2024a) and the broader tradition (Fahnestock, 2002; Dubremetz & Nivre, 2018) document extensively, figures such as *Anaphora*, *Epiphora*, and *Antimetabole*/*Chiasmus* have genuinely contested definitional boundaries even among expert rhetoricians — a challenge no amount of model scale resolves on its own.

We argue that these are not implementation bugs to be patched with a larger model, but structural mismatches between how figures work and how statistical models represent language. **iSocrates** responds to this mismatch with a hybrid architecture: symbolic, rule-based detection for figures with clear structural or phonetic signatures, paired with a language model for figures requiring contextual, semantic judgment — with each component deployed where it is actually suited, rather than defaulting to one method universally.

The system operates in two contexts within the broader Rhetoricon project (Harris & Di Marco, 2025): an **admin research tool** for exploratory analysis of texts and PDF documents, and **GoFigure**, a public crowdsourcing and moderation platform where community-submitted figure instances are automatically scored and validated against bibliographic authorities before human review.

Evaluated against 642 gold-standard instances spanning 163 figures — drawn from the same curated Rhetoricon *Doxa* database that underlies the group's broader ontological work (Wang, Berry & Harris, 2021; Wang, 2025) — iSocrates achieves 80.5% grand accuracy, compared to 16.6% for a single statistical classifier operating without symbolic support. We report this result together with an honest account of its limits (Section 6), and close by considering what this architecture suggests about the relationship between rhetorical theory and computational method more broadly (Section 7).

---

## 2. Why Statistical Models Struggle With Rhetorical Figures

### 2.1 The Single-Classifier Baseline
Our first approach used a standard fine-tuned classifier (DistilBERT; Sanh et al., 2019) to identify which figure, if any, was present in a given passage. Trained on 9,552 human-verified examples, this classifier reached only 16.6% accuracy across the full range of 163 figures. This is not a failure of implementation — it reflects a genuine structural limitation: a single-label classifier cannot represent the fact that passages frequently contain *multiple* overlapping figures simultaneously (a passage can be simultaneously alliterative, anaphoric, and antithetical), and cannot represent *where* within a passage a figure occurs.

### 2.2 The Deceptive Promise of Token-Level Tagging
We next attempted a more granular approach: tagging individual words as belonging to the beginning, interior, or exterior of a figure's span (a standard NLP technique called **BIO tagging**). This approach reported an apparently strong 87.3% accuracy. This number is misleading. Because the overwhelming majority of words in any passage belong to *no* figure at all, a model can achieve high raw accuracy simply by learning to say "no figure here" for nearly everything — while almost never correctly identifying an actual figure when one is present. When measured by a metric that actually credits correct figure identification (F1 score), this same model scored below 0.07 — essentially failing at the task it was ostensibly performing well at.

This divergence between the two numbers illustrates a caution relevant beyond computational method: an evaluation metric can look impressive while telling us almost nothing useful about performance on the phenomenon that actually matters.

We subsequently tested a more sophisticated model (DeBERTa; He et al., 2021) on the same token-level task, hoping its more advanced architecture would resolve the imbalance problem. In informal testing this approach proved numerically unstable and failed to train at all — a result we report as a design rationale rather than a rigorously logged experiment, but one consistent with our broader conclusion: the problem was not model sophistication, but a mismatch between the granular, token-level framing of the task and how these figures are actually structured.

### 2.3 Why Fragment-Level Judgment Fails
Both failed approaches share a common flaw: they ask a model to judge whether a *fragment* of text belongs to a figure, in isolation from the sentence's holistic structure. But most rhetorical figures are defined at the level of the whole utterance — the repetition across an entire clause boundary (*Anadiplosis*), the balance between two full phrases (*Antithesis*, *Isocolon*), the reversal of word order across a sentence (*Antimetabole*). Judging fragments in isolation cannot recover these relationships.

This led us to decouple two separate tasks: first, identifying *which* figures a passage likely contains (a holistic, sentence-level judgment, for which a lightweight classifier remains well suited), and second, *where exactly* within the passage the figure occurs and why (a task requiring the kind of contextual reasoning language models are comparatively better at, when properly guided).

---

## 3. A Hybrid Architecture: Matching Method to Figure

Rather than relying on any single method, iSocrates routes each candidate figure to whichever detection strategy actually matches its linguistic character:

- **Symbolic phonetic detection** for figures defined by sound (*Alliteration*, *Assonance*, *Consonance*), using the CMU Pronouncing Dictionary (Weide, 1998) to convert written words into their actual phonetic sound sequences (ARPAbet codes) — recovering exactly the sound-level information that subword tokenization destroys. When two words share a phonetic sound pattern, the system flags this deterministically, without needing the language model to "guess" at pronunciation from spelling.
- **Symbolic syntactic rules** for figures defined by structural repetition (*Epanaphora*, *Epiphora*, *Mesodiplosis*, *Isocolon*), following the tradition of ontological, rule-grounded figure detection established in this same research group (Wang, Berry & Harris, 2021; Wang, 2025; O'Reilly & Harris, 2017).
- **Language model verification**, guided by curated examples retrieved from the Rhetoricon *Doxa* database at the moment of judgment (a technique called Retrieval-Augmented Generation), for figures requiring genuine semantic or contextual reasoning — for instance, distinguishing true *Litotes* (understatement via double negation, e.g. *"not uninteresting"*) from simple literal negation (*"not interesting"*), a distinction requiring exactly the kind of unexcluded-middle reasoning discussed by Mitrović, O'Reilly, Harris & Granitzer (2020).

Only when a passage cannot be resolved by symbolic rules does the system fall back to language-model judgment — and even then, the model is not left to reason unaided: it is explicitly supplied with the classifier's candidate guesses, relevant prior human corrections, and canonical figure definitions before making a determination. Full technical detail on this pipeline is provided in the Implementation Notes appendix.

The system is deployed in two contexts: an interactive research tool allowing scholars to query passages or upload full documents for figure extraction, and **GoFigure**, a public platform where community contributors submit figure instances that are automatically scored and cross-checked against bibliographic authorities (the Library of Congress, Google Books, and Open Library, in that order of priority) before moderator approval.

---

## 4. Evaluation

We evaluated the complete hybrid pipeline against 642 gold-standard instances spanning all 163 figures in the Rhetoricon *Doxa* database, achieving **80.5% grand accuracy** (82.1% precision, 79.4% recall, 80.7% F1) — a substantial improvement over the single-classifier baseline (16.6%).

Performance varies considerably by figure type. Figures with strong structural or phonetic signatures perform best: *Erotema* (rhetorical questions) reached 90.0% accuracy; *Litotes*, *Metonymy*, *Oxymoron*, and several others reached 100% on their evaluated samples. Figures requiring finer syntactic discrimination — *Asyndeton* (43.3%), *Epanalepsis* (43.3%), *Epitrochasmus* (25.0%) — proved considerably harder, consistent with the genuine definitional and boundary difficulty these figures pose in the broader literature (Harris & Randhawa, 2025, on epanalepsis specifically; Kühn et al., 2024a, on boundary ambiguity generally).

Where comparable prior benchmarks exist in the literature (Kühn et al., 2024b), iSocrates' results compare favorably — for example, against Bhattasali et al.'s (2015) 53.7% F1 on rhetorical question detection, or Cho et al.'s (2017) 55.0% F1 on oxymoron detection. We present these as informative points of reference rather than controlled, head-to-head comparisons, since the underlying test sets differ; the full comparison table is provided in the Implementation Notes appendix.

### On Boundary-Disputed Figures
Several of our weaker-performing figures — *Antimetabole* (70.0%), *Ploke* (50.0%), *Epanalepsis* (43.3%) — are exactly the figures the field's own theoretical literature identifies as genuinely contested at the boundary (Kühn et al., 2024a, Section 4.3, on the recurring conflation of chiasmus and antimetabole; Fahnestock, 2002, on inconsistent classical taxonomy more broadly). We take this correlation as informative rather than incidental: where human rhetoricians themselves disagree about a figure's precise extent, a computational system built on curated examples of that same contested category should be expected to inherit some of that difficulty, rather than resolve it outright.

---

## 5. What This Suggests About Rhetorical Figures and Computation

The pattern of results across Section 4 suggests a broader claim relevant to computational rhetoric as a field: **the figures most resistant to statistical language models are not the most syntactically complex, but the most definitionally contested.** Purely phonetic or purely structural figures — however superficially "simple" — are handled well once the correct symbolic method is applied. What resists improvement even with a sophisticated language model in the loop are the figures where rhetoricians themselves have not settled on sharp boundaries (Section 4, above).

This has a methodological implication for the field beyond this system: closing the gap on these figures is unlikely to be a matter of better models or larger training sets. It is more likely to require the kind of definitional and ontological groundwork already underway within this research program (Wang, Berry & Harris, 2021; Wang, Kühn, Mitrović & Harris, 2022) — i.e., work that is properly rhetorical and linguistic in character, not primarily computational.

---

## 6. Limitations

We report the following limitations candidly:

- **Expert curation without a formal quantified agreement statistic.** Evaluation instances were verified through the Rhetoricon team's collaborative review process (Harris & Di Marco, 2025) rather than single-annotator labeling — a stronger provenance than uncontrolled or crowd-sourced data — but we did not compute a formal inter-annotator agreement statistic (e.g., Cohen's or Fleiss' kappa). We recommend this as a concrete next step, particularly for boundary-disputed figures, where even careful expert annotation within this same research lineage has previously reported disagreement (Strommer, 2011; Gawryjolek, 2009).
- **Uneven per-figure sample sizes**, ranging from 3 to 30 instances depending on how well-represented a figure is in the curated corpus; results for low-n figures should be read as indicative rather than statistically stable.
- **Non-normalized literature comparisons.** Comparisons to prior published benchmarks (Section 4) use each source's own reported metric and test set rather than a controlled, unified benchmark.
- **English-only scope**, consistent with a limitation the field broadly shares (Kühn et al., 2024a, 2024b), though multilingual ontological groundwork already exists within this research tradition (Wang, Kühn, Mitrović & Harris, 2022) and is a natural next extension.
- **Single-configuration evaluation.** All results reflect a single deployed system configuration; sensitivity to alternative implementation choices was not systematically tested.

---

## 7. Conclusion

The computational neglect of most rhetorical figures, documented extensively by Kühn et al. (2024a, 2024b), is not simply a matter of insufficient research attention — it reflects a genuine mismatch between how contemporary language models represent text and how most rhetorical figures actually function linguistically. iSocrates demonstrates that a hybrid approach, matching detection method to a figure's actual linguistic character rather than defaulting to a single statistical technique, can meaningfully close this gap for a broad range of previously unaddressed figures. What remains resistant — genuinely boundary-disputed figures — points toward a conclusion we think is important for computational rhetoric as a field: progress here depends as much on continued rhetorical and ontological scholarship as on further advances in machine learning.

---

## Appendix A: Implementation Notes

*This appendix summarizes the technical architecture, training methodology, and infrastructure decisions underlying the system described above, for readers interested in implementation detail.*

**Architecture summary.** The system implements a three-tier "fast-path" pipeline: (1) an exact-match database lookup against previously verified instances; (2) deterministic symbolic rules for phonetic figures (via CMU Pronouncing Dictionary phoneme lookup) and structural repetition figures (via syntactic pattern matching), each resolved in under 1 millisecond; (3) a locally-hosted 7-billion-parameter language model (Qwen 2.5, 4-bit quantized GGUF format, run via `llama.cpp`) for figures requiring semantic judgment, guided by Retrieval-Augmented Generation prompts drawing curated examples from the Doxa database.

**Model selection.** Gemma 2B and Phi-3-mini were informally evaluated as lighter-weight alternatives; both were deprioritized in favor of Qwen 2.5 7B not on accuracy grounds but for deployment reasons — open licensing and a 4.3 GB memory footprint fitting within a strict 6 GB budget for fully local, privacy-preserving, rate-limit-free operation.

**Training corpus.** 22,699 human-verified annotations across 163 figure classes, split 80/10/10 (train/validation/test) via stratified sampling; figures with fewer than 3 training instances rely on few-shot prompting with canonical definitions rather than fine-tuning.

**Human-in-the-loop correction.** Moderator corrections in the public GoFigure platform are written back to the system's database and dynamically injected into subsequent language model prompts, allowing the system to avoid repeating known errors within and across sessions.

**Defensive engineering.** Given the system processes uncurated, user-submitted text, the implementation includes structured-output enforcement (constraining language model output to valid JSON), boundary-guard checks against malformed phonetic input, and cross-language (Python/Go) consistency fixes for Unicode text handling — details available on request but omitted here as outside the paper's rhetorical scope.

**Full per-figure evaluation results, training logs, and literature comparison tables** are available in the companion technical report (`iSocrates_Full_Technical_v4.md`) and can be provided to reviewers on request.

---

## References
1. Kühn, R., Mitrović, J., & Granitzer, M. (2024a). *The Elephant in the Room: Ten Challenges of Computational Detection of Rhetorical Figures*. Proceedings of the 4th Workshop on Figurative Language Processing (FLP), 45–52.
2. Kühn, R., Mitrović, J., & Granitzer, M. (2024b). *Computational Approaches to the Detection of Lesser-Known Rhetorical Figures: A Systematic Survey and Research Challenges*. ACM Computing Surveys. arXiv:2406.16674.
3. Fahnestock, J. (2002). *Rhetorical Figures in Science*. Oxford University Press.
4. Burton, G. O. (2007). *Silva Rhetoricae: The Forest of Rhetoric*. Brigham Young University.
5. Dubremetz, M., & Nivre, J. (2018). *Rhetorical figure detection: chiasmus, epanaphora, epiphora*. Frontiers in Digital Humanities, 5, 10.
6. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT*. arXiv preprint arXiv:1910.01108.
7. He, P., Gao, J., & Chen, W. (2021). *DeBERTaV3*. arXiv preprint arXiv:2111.09543.
8. Weide, R. L. (1998). *The Carnegie Mellon Pronouncing Dictionary*. CMU Speech Group.
9. Mitrović, J., O'Reilly, C., Harris, R. A., & Granitzer, M. (2020). *Cognitive modeling in computational rhetoric: Litotes, containment and the unexcluded middle*. ICAART, Valletta, Malta.
10. Bhattasali, S., Cytryn, J., Feldman, E., & Park, J. (2015). *Automatic identification of rhetorical questions*. ACL/IJCNLP 2015 (Vol. 2), 743–749.
11. Cho, W. I., Kang, W. H., Lee, H. S., & Kim, N. S. (2017). *Detecting oxymoron in a single statement*. O-COCOSDA 2017, 1–5.
12. Paida, K. (2019). *Double Negation Detection* [Master's thesis, University of Passau].
13. Wang, H., Du, S., Zheng, X., & Meng, L. (2023). *An empirical study of incorporating syntactic constraints into BERT-based location metonymy resolution*. Natural Language Engineering, 29(3), 669–692.
14. Kühn, R., Saadi, K., Mitrović, J., & Granitzer, M. (2024). *Using pre-trained language models in an end-to-end pipeline for antithesis detection*. LREC 2024. ELRA.
15. Harris, R. A., & Di Marco, C. (2025). *The Rhetoricon Database: An overview and an appreciation*. CMNA XXV, Online, 12 December.
16. Harris, R. A., & Randhawa, Z. (2025). *Epanalepsis in argumentation: Pseudo tautologies*. CMNA XXV, Online, 12 December.
17. O'Reilly, C., & Harris, R. A. (2017). *Antimetabole and image schemata: Ontological and vector space models*. JOWO 2017, Bozen-Bolzano, Italy.
18. Wang, Y., Berry, D., & Harris, R. A. (2021). *An ontology for ploke: Rhetorical figures of lexical repetition*. Joint Ontology Workshops 2021, Bozen-Bolzano.
19. Wang, Y. (2025). *A knowledge representation for, and an application to requirements elicitation of, rhetorical figures of perfect lexical repetition* [Doctoral dissertation, University of Waterloo].
20. Wang, Y., Kühn, R., Mitrović, J., & Harris, R. A. (2022). *Towards a unified multilingual ontology for rhetorical figures*. KEOD 22, Valletta, Malta.
21. Kampherm, M. (2023). *Masks and caricatures: Prosopopoeia, ethopoeia, and the effect of social media on Canadian political leaders' debates* [Doctoral dissertation, University of Waterloo].
22. Tu, K. (2019). *Collocation in rhetorical figures: A case study in parison, epanaphora and homoioptoton* [Major research paper, University of Waterloo].
23. Strommer, C. W. (2011). *Pursuing the relations between saliency, context, intent, and rhetorical figures in text media* [Doctoral dissertation, University of Waterloo].
24. Gawryjolek, J. (2009). *Automated annotation and visualization of rhetorical figures* [Master's thesis, University of Waterloo].
