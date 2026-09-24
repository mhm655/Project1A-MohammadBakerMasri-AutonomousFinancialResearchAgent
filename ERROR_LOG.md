# ERROR_LOG — Deliberate Errors in the Project Document

The project document says it contains **exactly 7 deliberate factual or logical errors** across Parts
A–E (Section A4.3, "Error Detection Exercise"). Below are the seven I identified, each with the
location, the erroneous text, why it is wrong, and the correction. Two further candidates follow at
the end, in case any of the seven turns out to be unintended.

> Status: identified on Day 1 (first full read). To be re-checked on Day 14 before submission.

---

## Error 1: AB-4 Memory Utilization formula uses multiplication instead of division

- **Location:** Part A, Section A5.2, Category 5, metric **AB-4**.
- **Text:** *"Measured as the ratio of memory hits to total external API calls (higher is better,
  target: >=0.3). Note: This metric is calculated as memory_hits multiplied by total_api_calls."*
- **Why it is wrong:** The metric is described as a **ratio** with a target of ≥ 0.3. A product
  `memory_hits × total_api_calls` is not a ratio. It has no fixed scale, grows without bound, and
  *increases* as the agent makes more external calls, which is the opposite of what the metric is
  meant to reward. A 0.3 target only makes sense for a quotient.
- **Correction:** `AB-4 = memory_hits / total_external_api_calls`. ARA-1 implements it this way in
  `evaluation/metrics.py`.

## Error 2: The source reliability hierarchy ranks social media above major news outlets

- **Location:** Part A, Section A6.2, "Tiered Source Trustworthiness".
- **Text:** *"Tier 4: Social media posts and anonymous forum discussions – crowd-sourced but
  unverified and subject to manipulation."* followed by *"Tier 5: Major news outlets (Reuters,
  Bloomberg News, Financial Times) – professional journalism with editorial oversight…"*
- **Why it is wrong:** The list is ordered from highest to lowest reliability. By its own
  descriptions, unverified, manipulable, anonymous content is less reliable than journalism with
  editorial oversight. With this ordering, the "apply the highest-tier rule" step in A6.3 would pick
  a Reddit post over Reuters in a conflict.
- **Correction:** Swap them. Tier 4 is major news outlets; Tier 5 is social media and anonymous
  forums. ARA-1 uses the corrected order (architecture spec §12.3).

## Error 3: Wrong date and causality for the first US bank stress tests (SCAP)

- **Location:** Part A, Section A7.3, "Query Disambiguation".
- **Text:** *"A query about 'bank stress tests' in 2007 likely refers to the European Banking
  Authority's stress test programme … Note: The first US bank stress tests under SCAP were conducted
  in 2007 following the Dodd-Frank Act."*
- **Why it is wrong:**
  - The Supervisory Capital Assessment Program (SCAP) was conducted by the Federal Reserve in
    **2009**. The paragraph itself says this one sentence earlier, so the document contradicts itself.
  - The **Dodd-Frank Act was signed in July 2010**, after SCAP, so SCAP cannot have followed it.
    Dodd-Frank later made stress testing a statutory requirement (DFAST), and the Fed's CCAR began in
    2011.
  - The **European Banking Authority was only established in January 2011**, so no "EBA stress test
    programme" existed in 2007. The first EU-wide stress test, in 2009, was run by its predecessor,
    CEBS.
- **Correction:** SCAP was the first US supervisory stress test, conducted in 2009 in response to
  the financial crisis. Dodd-Frank (2010) came afterwards and institutionalised annual stress tests.

## Error 4: Wrong embedding dimension for `text-embedding-3-large`

- **Location:** Part E, Section E2.2, "Recommended Embedding Models".
- **Text:** *"OpenAI text-embedding-3-large: 1024 dimensions, $0.13 per million tokens."*
- **Why it is wrong:** `text-embedding-3-large` outputs **3072**-dimensional vectors by default. (It
  can be shortened through the `dimensions` parameter, but its native size is 3072.) As written, the
  "large" model would have *fewer* dimensions than `text-embedding-3-small` (1536), which is
  implausible on its face. The error matters in practice: a vector index created for 1024 dimensions
  would reject 3072-dimensional embeddings.
- **Correction:** 3072 dimensions (native). The price of $0.13 per million tokens, and the "6.5×"
  cost multiple over the small model's $0.02, are consistent.

## Error 5: Form 20-F attributed to Indian filings with the Ministry of Corporate Affairs

- **Location:** Part C, Case Study 4, Section C4.2, "Filing format differences".
- **Text:** *"Indian companies file annual returns using Form 20-F with the MCA (Ministry of
  Corporate Affairs), similar to the 10-K filing in the US system."*
- **Why it is wrong:** **Form 20-F is a US SEC form.** It is the annual report filed with the SEC by
  *foreign private issuers* with US-listed securities (for example, Infosys and HDFC Bank file 20-F
  because of their NYSE-listed ADRs). It has nothing to do with the MCA. Indian companies file their
  annual return with the MCA on **Form MGT-7 / MGT-7A** and their financial statements on
  **Form AOC-4**. Listed companies also publish annual reports through the BSE and NSE under SEBI
  LODR regulations.
- **Correction:** Indian companies file MGT-7 (annual return) and AOC-4 (financial statements) with
  the MCA. Form 20-F is filed with the US SEC only by foreign issuers listed in the US.

## Error 6: Unsupported claim of a 45–60% "industry average" hallucination rate

- **Location:** Part C, Case Study 3, Section C3.2, "What Went Wrong".
- **Text:** *"Hallucination rate of 23% … Note: Industry average hallucination rates for unverified
  financial agents are typically around 45-60%."*
- **Why it is wrong:** No such industry average exists, and the note undermines the case study's own
  argument. The case presents 23% as a failure that justified a complete redesign. If the norm were
  45–60%, then 23% would be *better than half the industry average*, and the "failed first attempt"
  would be a success. The sentence contradicts its context, and the figure is unsourced. (Measured
  hallucination rates depend heavily on the task and the definition used; they cannot be summarised
  as one "industry average".)
- **Correction:** Delete the note. The case study is internally consistent without it: 23%
  unverified, falling to 1.2% after adding verification.

## Error 7: The progression unlocks contradict the tools each challenge expects

- **Location:** Part B, Section B4.4 ("Progression Unlocks"), compared with Section B2.1 (challenge
  definitions).
- **Text (B4.4):** *"After Challenge 2: The agent gains access to the earnings_transcript tool …
  After Challenge 4: … the peer_comparison tool … After Challenge 6: … the calculation_engine tool …"*
- **Why it is wrong:** The challenge definitions require these tools *before* they are unlocked:
  - Challenge 2 (earnings analysis) lists `earnings_transcript` as an expected tool and is judged on
    the "quality of transcript analysis", yet the tool is only unlocked *after* Challenge 2.
    Challenge 3 also expects it.
  - Challenge 4 lists both `peer_comparison` and `calculation_engine` as expected tools, but they
    unlock after Challenges 4 and 6 respectively.

  An agent that follows B4.4 cannot meet the B2.1 expectations for Challenges 2, 3 and 4. The
  "Full Stack Badge" (all 12 tools used across the 8 challenges) is also harder to earn than intended.
- **Correction:** Either unlock each tool before the first challenge that expects it
  (`earnings_transcript` before C2, `peer_comparison` and `calculation_engine` before C4), or remove
  those tools from the expected-tool lists of the earlier challenges. ARA-1 implements the unlock
  mechanism as specified (a config-driven `UnlockRule`, with no code changes), but its default
  configuration unlocks every tool from the start.

---

## Additional candidates (lower confidence)

These are also inaccurate, but they are less likely to be the *deliberate* errors:

- **B3.1, "five dimensions":** *"Each challenge is scored across five dimensions"*, but the scoring
  matrix has only **four** (Accuracy, Completeness, Insight, Architecture) plus a Total column. The
  row and column totals are otherwise arithmetically correct (the columns sum to 205 + 165 + 190 +
  140 = 700).
- **A6.2, Tier 1 described as "audited":** *"SEC filings (10-K, 10-Q) – legally mandated,
  audited…"*. Financial statements in a **10-Q are reviewed, not audited**. Only the 10-K contains
  audited statements, and the document's own glossary says the 10-Q "contains unaudited financial
  statements".
