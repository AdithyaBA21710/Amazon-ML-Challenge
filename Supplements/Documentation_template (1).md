# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TODO: Your Team Name]  
**Team Members:** [TODO: List all team members]  
**Submission Date:** [TODO: Date]

---

## 1. Executive Summary

We solve the task as a two-stage pipeline: a multi-channel TF-IDF blocking stage that retrieves, for every Source 2/3 record, a short list of possible Source 1 owners within the same country, followed by a LightGBM classifier over 51 pair features and a decision rule tuned directly on macro F0.5. The key ideas are (1) reversing the search direction and enforcing the "each S2/S3 record has at most one owner" property we found in the training data, (2) a dictionary learned only from the training labels that translates Indian-script business names into English, (3) word-distinctiveness features that separate noisy copies of a business from genuinely different sibling businesses, and (4) **training and validating at full density**, with every training business present in the candidate pool exactly as in the test set. On held-out training entities under realistic conditions, the final model scores **0.9620 macro F0.5**, up from 0.9383 for the same features trained on a sparse sample.

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale.** Training contains 2,206,821 Source 1 entities and 10,320,219 Source 2/3 records (5,034,616 + 5,285,603); the test set contains 1,732,544 Source 1 entities and 9,969,589 Source 2/3 records. Brute-force comparison is impossible, so memory-efficient blocking and batch processing were design requirements from the start.

**Structure of the labels** (all measured on the full training ground truth):

| Finding | Value | Consequence for the design |
|---|---|---|
| Singleton rate (S1 entities with no match) | 5.6% | "Predict nothing" scores only 0.056; recall matters as well as precision |
| Matches per entity | mostly 2 to 6, max 11 (mean about 3.5) | Missing one of several matches lowers per-entity recall only partly |
| S2/S3 records owned by more than one S1 entity | **0** out of 7,638,365 matches | **One-owner rule**: each S2/S3 record is assigned to at most one S1 entity |
| Matched pairs that cross countries | **0** | Blocking is done within each country label |
| S2/S3 records with an owner | about 74% | The other 26% are distractors that must stay unmatched |
| Empty addresses | 0 in S1; about 3.4% in S2/S3 | "Missing address" is an explicit feature |

**Noise patterns observed** (from reading hundreds of matched pairs):

- **Names:** word-order shuffles ("of Group Bnp Companies-Delhi"); names turned into domains or handles ("bnpgroupcompaniesdelhi.com", "@jexfirst"); scrambled characters ("Ttuaesb" for "Tubes"); typos inside legal suffixes ("Pdivate", "Parien"); former and trade names ("Irisyn One formerly: Tirth Resources…"); completely different trade names at the same address ("Dovacira" for "Cab Mills"); digits used as letters ("C0mmunity"); "The" in any position ("Dent Seafood The"); generic words appended or swapped ("… Inc Enterprises", "… Partners").
- **Indian scripts:** 752,869 training S2/S3 names (and 867,245 test names) are written fully or partly in Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Odia or Malayalam, e.g. "சன் டெக் பிரைவேட் லிமிடெட்" for "Sun Tech Private Limited". A naive ASCII normalizer turns these into empty strings.
- **Addresses:** abbreviations (St/Street, and even "Saint" for "St"), component reordering, missing components, zero-padded house numbers ("003808"), small perturbations of house numbers in true matches ("27/2" vs "28/2"), literal "null" tokens, state names in native script or as codes ("தமிழ்நாடு", "TN").
- **Sibling businesses:** different companies sharing most of a name and a nearby address ("Urban Constructions" vs "Urban Developers", No. 443 vs 444). These are rare in a small sample but common in the full pool, which turned out to matter a great deal (Section 2.3).
- **Test-only country:** France appears only in the test set (259,452 S1 entities, about 15% of the test set).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (with metric-aware post-processing)  
**Core Innovation:** A reversed, one-owner formulation (each S2/S3 record searches for its single owner in Source 1), combined with a training-label-derived Indian-script dictionary, IDF-based "distinctive word mismatch" features, and a full-density training and validation protocol that reproduces the crowding of the real test set.

Pipeline:

1. **Normalize** every record once (country-agnostic rules) and store as Parquet.
2. **Learn** an Indian-script → English word dictionary from the training labels.
3. **Block:** three TF-IDF nearest-neighbour channels per country, S2/S3 → S1, top 10 each, against the **full** pool of Source 1 businesses.
4. **Featurize** each candidate pair (51 features).
5. **Classify** with LightGBM trained on full-density candidates from half A of the training entities.
6. **Decide:** keep each S2/S3 record's most likely owner only, and accept it if its probability passes a threshold chosen to maximize macro F0.5 on half B.

**Country handling.** Country is treated as an open set of labels: every stage loops over whatever labels occur (so France is processed exactly like the training countries), no rule is specific to US or India, and no feature uses the country value.

### 2.3 Validation Design, and What We Learned

We first developed on a 150,000-entity sample of the training data (all true matches of the sampled entities plus a proportional share of distractor records). Out-of-fold, it scored 0.9877, but our first leaderboard submission scored **0.902**.

To find out why, we ran the full pipeline on the **entire** training set and scored it against the training labels (excluding the sampled entities). It scored **0.9383**. The sample contained only 7% of the businesses, so most lookalikes of each business were missing: the true owner nearly always stood out, and the model learned to rely on that ("the best candidate is clearly ahead" features carried 62% of the gain). In the full pool, siblings and lookalikes are everywhere, and that shortcut fails. At full density, 12.9% of entities with matches received at least one wrong ID, and only 80.9% of singletons were correctly left empty.

**Final protocol.** Training Source 1 entities are split into two fixed halves by a CRC32 hash of their id. Each S2/S3 record belongs to its owner's half (unowned records are hashed on their own id).

- **Half A:** blocking runs against the full training pool; features are built for a random 20% of half-A records, with complete candidate lists. The model is trained on these.
- **Half B:** never trained on. The whole training set is scored with the final model, and half-B entities are used to choose the threshold and to report the realistic score. A record assigned to a half-B entity counts against it even if its true owner is in half A, exactly as on the leaderboard.

The remaining gap between the realistic score and the leaderboard is attributable mainly to France, which has no labels in training (Section 5).

---

## 3. Candidate Generation (Blocking)

**Normalization (before blocking).** Accent stripping, lower-casing, "&" → "and", punctuation removal; splitting of "formerly / dba / aka / trading as" names into a main and an alternate name; removal of web decorations ("www.", "@", ".com") with a web-name flag; removal of legal suffixes including French forms (SARL, SAS, SA, EURL…) and fuzzy-matched typos of long legal words; digit-for-letter repair inside words; removal of "the"; a spaces-removed form of the name; address abbreviation canonicalization (St/Street/Saint → one token, Rd/Road, Ave, Blvd, Nr/Near, Opp…); zero-stripped house numbers; "null" tokens treated as empty; and translation of Indian-script words via the learned dictionary.

**Learned Indian-script dictionary.** For training pairs where the S2/S3 name contains Indian script and has the same number of words as its owner's English name, words are aligned by position and counted. A script word's most frequent English partner is kept if it was seen at least 3 times and accounts for at least half of that word's alignments. This produced **1,347 entries** from 513,751 aligned pairs (e.g. लिमिटेड → limited, प्राइवेट → private, प्रा → pvt, லிமிடெட் → limited), covering **96.4% of the Indian-script words** in the test set (86.0% of such names fully translated). It uses only the provided training labels; no external transliteration or translation service is involved.

**Blocking keys used.** Three TF-IDF channels, each an index over the Source 1 records of one country, queried by that country's Source 2/3 records:

| Channel | Text indexed | Tokens | Purpose |
|---|---|---|---|
| name | core name + alternate name | words | standard name matches, robust to word order |
| addr | normalized address | words and numbers | finds trade names and renamed businesses via the address |
| char | name without spaces | character 3-grams | concatenated names, domains, handles, typos |

Tokens that occur in more than 0.5% (name, addr) or 1% (char) of a country's Source 1 records are dropped from the index: they carry little identifying information and dominate search cost. Sparse matrix products are computed in batches, in parallel, and each channel keeps the top **k = 10** Source 1 candidates per query. The union of the three channels forms the candidate set.

**Candidate pairs generated.** Full training set: 107,796,669 pairs for India (4,133,346 records) and 160,381,769 for the US (6,186,873 records), about 26 per record. Test set: [TODO: total test candidate pairs]; only 861 of the 1,732,544 test Source 1 entities (0.05%) receive no candidates at all.

**How we ensured true matches were not lost.** Every blocking change was measured against the ground truth by pair recall per channel and per country, and by the **ceiling score**: the macro F0.5 a perfect classifier would reach on the candidate set. On the 150k development sample:

| k (per channel) | name | addr | char | **Union** |
|---|---|---|---|---|
| 1 | 52.58% | 85.21% | 64.93% | 95.35% |
| 3 | 62.51% | 88.70% | 75.55% | 97.64% |
| 5 | 65.64% | 89.53% | 79.95% | 98.32% |
| 10 | 68.53% | 90.48% | 84.88% | **99.00%** |

No single channel exceeds 91% recall, but their union reaches 99.00%. Adding the learned script dictionary raised union recall from 98.21% to 99.00% and India's from 96.26% to 98.20%. At full density the task is harder: the ceiling on the full training set is **0.9816** (sample: 0.9963), and the true owner is among the candidates for 92.2% of owned half-A India records, since more lookalikes compete for the top 10.

`output/candidate_pairs.tsv` is exactly the set of pairs the model scored on the test set, grouped by Source 1 entity.

---

## 4. Matching Model

**Features used** (51 in total; all country-agnostic):

- **Name features:** Levenshtein-based ratio, token-sort ratio, token-set ratio, partial ratio, Jaro-Winkler; the same measures on the spaces-removed name (catches "jexfirst" vs "jex first"); similarity of sorted characters (catches scrambles); token-set ratio on the full name including legal words; best match against alternate ("formerly/dba") names; first-word equality; word counts and their difference; web-name flag; Indian-script flag.
- **Word-distinctiveness features:** each word's IDF over Source 1 names (capped at 10 so values are comparable across pools of different sizes). For each pair, words are aligned (allowing small typos), and we compute the share of each name's total IDF that is unmatched, the IDF of the most distinctive unmatched word on each side, the number of unmatched words, and the total IDF of matched words. An extra "Services" or "Partners" barely registers, while "Developers" in place of "Constructions" scores high.
- **Address features:** ratio, token-set, token-sort and partial ratios; missing-address flag; address length; house-number overlap, Jaccard, conflict and first-number equality; log absolute difference and digit-string similarity of the first house number.
- **Blocking features:** similarity score and rank from each blocking channel (missing when a channel did not retrieve the pair).
- **Context features:** each pair compared with the other candidates of the same S2/S3 record: number of candidates, number of records that retrieved this S1 entity (counted over the whole country), gap to the best candidate on name, address, character and combined similarity, rank on combined similarity, and the margin over the runner-up.

**Model type:** LightGBM gradient-boosted trees (MIT license), binary objective. Parameters: learning rate 0.05, 255 leaves, minimum 200 samples per leaf, feature fraction 0.85, bagging fraction 0.8, L2 regularization 1.0. Training data: all candidate pairs of a random 20% of half-A records at full density (about 27 million pairs; all positives and 50% of negatives used). Early stopping on 10% held-out records selected 703 trees; the final model is retrained on all sampled pairs with 773 trees. Held-out AUC is 0.99995. No pretrained models, external APIs or external data are used.

**Threshold selection method:** direct optimization of macro F0.5 on half B, after applying the one-owner rule (each S2/S3 record keeps only its highest-probability candidate). We sweep thresholds from 0.30 to 0.98 in steps of 0.02, then refine in steps of 0.005 around the best value. The selected threshold is **0.845**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9620** on half B of the training entities at full density (realistic protocol, Section 2.3). Public leaderboard: [TODO: v2 score] (first submission with the sample-trained model: 0.902).

| Version | Evaluation | Macro F0.5 | Precision | Recall | India | US | Singletons correct |
|---|---|---|---|---|---|---|---|
| v1: sample-trained | 150k sample (optimistic) | 0.9877 | 99.56% | 97.56% | 0.9831 | 0.9908 | 97.79% |
| v1: sample-trained | full training set | 0.9383 | 95.96% | 93.47% | 0.9105 | 0.9568 | 80.9% |
| **v2: full-density trained** | **half B, full density** | **0.9620** | **99.13%** | **92.32%** | **0.9409** | **0.9761** | **95.9%** |

Where points are lost (out of 1.0, at full density):

| Cause | v1 | v2 |
|---|---|---|
| Singletons given a match | 0.0107 | 0.0023 |
| Entities with at least one wrong ID | 0.0283 | 0.0060 |
| Entities only missing matches | 0.0228 | 0.0297 |

Training at full density cut the cost of wrong merges by about 80%. The model now trades a little recall for much higher precision, which F0.5 rewards. The remaining loss is dominated by missed matches, part of which is the blocking ceiling (0.9816).

**France.** France has no labels, so its quality can only be inferred from the leaderboard. Using the v1 full-density scores for India and the US weighted by their test shares, the v1 leaderboard score of 0.902 implies roughly 0.73 for France. [TODO: update with the v2 leaderboard score.]

- **Common false positives (wrong merges):**
  - *Sibling businesses:* different companies sharing most of a name and a nearby address, e.g. "Urban Constructions" vs "Urban Developers" at No. 443 vs 444, or "Bay Council" at 15740 vs 15751 Broadway. Full-density training and the word-distinctiveness and house-number features reduced these substantially.
  - *Unrelated names at a known address:* an S2/S3 record with an unrelated name at an S1 entity's address. Genuine trade names look identical, so these are inherently ambiguous.
  - *Near-identical names with identical addresses* that the labels treat as different businesses ("Ujjansh" vs "Ujjinsh").
  - *Information removed by normalization:* "Private Limited" vs "Public Limited" become identical once legal suffixes are stripped; Indian-script words missing from the dictionary (e.g. Odia for "Automobiles") disappear.
- **Common false negatives (missed matches):**
  - True owners pushed out of the top 10 candidates by lookalikes in the full pool (blocking ceiling 0.9816).
  - Short names with no address whose one distinguishing word was changed ("Slate" vs "Slate Ventures", "Pediatric Safe Care" vs "Pediatric Safe Enterprises"); without an address, the model stays below the precision-oriented threshold.
  - Heavily misspelled names with no address, names written as initials or domains ("cndelta.com"), and numbers written as words ("FOURTEENTH" vs "14th").

---

## 6. Conclusion

A two-stage pipeline (recall-oriented multi-channel blocking, then a gradient-boosted classifier with metric-aware decisions) reaches 0.9620 macro F0.5 under realistic, full-density conditions. The largest gains came from understanding the data rather than from model complexity: the one-owner property, learning Indian-script vocabulary from the labels themselves, features that capture *which* words differ rather than *how many*, and, above all, making training and validation match the density of the test set. Our most important lesson: a validation sample that removes most of the lookalike businesses gives a misleadingly high score (0.988 vs 0.938 realistic) and teaches the model shortcuts that fail at full scale. Validation must reproduce the conditions of the test set, not just its label distribution.

---

## Appendix

### A. Code Artefacts

The complete pipeline is in `code/business_entity_resolution/`:

```
business_entity_resolution/
├── run_all.sh          # single entry point: raw data -> both output files
├── README.md           # environment setup and exact run instructions
├── requirements.txt    # pinned library versions
└── src/
    ├── config.py       # DATA_DIR: folder containing train/ and test/
    ├── normalize.py    # text normalization rules (names, addresses, script dictionary)
    ├── preprocess.py   # normalize all sources -> Parquet (also builds the 150k sample)
    ├── script_dict.py  # learn Indian-script -> English dictionary from training labels
    ├── blocking.py     # 3-channel TF-IDF blocking (library + sample recall report)
    ├── features.py     # pair features (library + sample feature builder)
    ├── build_dense.py  # full-density blocking on train; features for half-A records
    ├── train_dense.py  # train LightGBM on full-density features
    ├── predict.py      # blocking + features + model + decision -> TSVs (train or test)
    ├── tune.py         # threshold and realistic score on half B
    ├── score.py        # score any prediction file against training labels (dev tool)
    ├── train.py        # first model on the 150k sample (dev tool, superseded)
    └── data.py         # exploratory analysis (dev tool)
```

**Reproduce:** `bash run_all.sh /path/to/dataset`. This runs: preprocess (train) → script_dict → preprocess (train + test, with dictionary) → build_dense → train_dense → predict (train) → tune → predict (test), writing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

**Runtime** on an AWS r6i.4xlarge (16 vCPU, 128 GB RAM): full-density blocking and features on train about 50 minutes; training about 8 minutes; scoring the training set about 40 minutes; test prediction about 75 minutes. Seeds are fixed at 42.

### B. Additional Results

**Effect of the Indian-script dictionary on blocking (150k sample):**

| | Before | After |
|---|---|---|
| Union recall | 98.21% | 99.00% |
| India recall | 96.26% | 98.20% |
| Missed true pairs | 9,281 | 5,205 |
| Ceiling macro F0.5 | 0.9929 | 0.9963 |

**Feature importance shift.** Trained on the sparse sample, the model put 62.0% of its gain on a single context feature (combo_margin). Trained at full density: combo_margin 38.9%, combo_gap 33.2%, combo_rank 15.2%, num_jacc 3.3%, a_tset_gap 1.1%, w_unmatched_n_b 1.0%, num_first_logdiff 0.9%, w_unmatched_max_b 0.8%, num_first_digit_sim 0.7%. House-number and word-distinctiveness features gained weight, as expected when siblings must be told apart.

**Held-out records at full density (best candidate per record):**

| Threshold | Precision | Recall |
|---|---|---|
| 0.5 | 97.40% | 98.23% |
| 0.7 | 98.50% | 97.37% |
| 0.9 | 99.34% | 95.73% |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
