# Plea-bargaining reform and the sentencing gap for non-local defendants in China

**Yiling (Elaine) Long** · working paper in progress

China's *leniency for confession and acceptance of punishment* reform (认罪认罚从宽) was
piloted in 18 cities from November 2016 and rolled out nationwide in October 2018, following
an earlier fast-track sentencing pilot (速裁, from August 2014). Non-local defendants —
people tried outside the city where they are registered — are less likely to receive
probation or bail than local defendants, in part because courts see them as having weaker
community ties. This project asks whether the reform narrowed or widened that gap.

> **Status (October 2026).** Data pipeline, measurement and a pre-registered main
> specification are complete; human validation of the LLM-extracted variables is under way.
> All estimates below are preliminary. No paper draft is posted here.

---

## Design

- **Data.** About one million Chinese criminal court judgments (Zhang, Kwan & Fang, 2024).
  Main sample: first-instance theft judgments dated 2013–2019, first-listed defendant —
  380,615 judgments from 358 cities, 18 of them pilot cities.
- **Identification.** Triple difference: pilot city × reform period × non-local defendant,
  with city × non-local and quarter × non-local fixed effects; separate coefficients for the
  fast-track pilot, the plea-reform pilot and the national rollout. Event-study estimates by
  quarter, Callaway–Sant'Anna as a staggered-adoption check.
- **Outcomes.** Probation; on bail at judgment; log sentence length; and, as the first stage,
  whether any leniency procedure was applied.
- **Inference.** Standard errors clustered by city, wild cluster bootstrap (9,999 draws), and
  Romano–Wolf correction across the four outcomes × two parameters of interest.
- **Pre-registration.** The main specification, controls, robustness checks and placebo
  tests were fixed in a pre-analysis plan before any estimate of the parameter of interest
  was seen; later additions are logged and labelled exploratory.

## Measurement: rules first, two LLMs for the hard cases

Defendant background (registered origin, local residence), pre-trial measures and procedure
are buried in unstructured judgment text.

1. A rule-based extractor (`src/extract_rules.py`, `src/segment.py`, `src/gazetteer.py`)
   parses each judgment and resolves place names to prefecture level with an administrative
   gazetteer.
2. Judgments the rules cannot resolve are sent to two independent LLMs (DeepSeek and Gemini;
   prompts in `prompts/`); disagreements go to an adjudication step (`src/llm_extract.py`).
3. A stratified random sample is hand-coded as a gold standard to measure error rates and to
   correct estimates for measurement error.

Total LLM cost for the full corpus was under USD 50. Unit tests in `tests/` cover the
parsers, the gazetteer, the LLM interface and the estimation code.

## Preliminary results

- The first stage is strong: in pilot cities the share of cases handled under a leniency
  procedure rises sharply after each pilot begins.
- The pre-registered estimates of the change in the non-local–local gap are small and
  statistically insignificant for all four outcomes, and stable across the robustness checks.
- With current standard errors, the minimum detectable effect is about half of the baseline
  non-local–local gap in probation and bail, so the design can rule out large changes but not
  moderate ones.
- Exploratory: in the baseline period, pre-trial detention accounts for most of the raw gap
  in probation (about 57%) and nearly all of the gap in bail.

## Repository layout

```
src/        reusable modules: segmentation, rule extraction, gazetteer, treatment
            assignment, LLM extraction, estimation
scripts/    one script per pipeline step; the prefix (T01–T09) is the step number
tests/      pytest unit tests
prompts/    LLM extraction and adjudication prompts (in Chinese, as applied to the text)
output/     figures and aggregate tables (main results: output/tables/T09b_*)
```

## Data and reproduction

Judgment texts and document-level derived data are **not** redistributed here. The source
dataset is public:

> Zhang, Y., Kwan, M.-P., & Fang, L. (2024). An LLM driven dataset on the spatiotemporal
> distributions of street and neighborhood crime in China. *Scientific Data*.

Paths to the raw files are set in `src/paths.py`. Python ≥ 3.11; install with
`pip install -r requirements.txt`; run the tests with `pytest`. LLM steps need API
credentials in a local `.env` file, which is not part of the repository.
