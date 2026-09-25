# LLM Rubric Evaluation Harness

A harness for grading language model responses against written rubrics instead of
against a single reference string, and for measuring how much the grading itself can be
trusted.

A rubric is a weighted set of criteria, each with five written level anchors. A response
is scored on every criterion, the criterion scores are combined by weight, and the result
is compared with a pass threshold. Two kinds of grader do the scoring: an objective
grader for numbers and required facts, and a model acting as a judge for the criteria
that need judgement. The harness also measures agreement between two judges, because a
rubric score is only usable if a second grader reading the same anchors would land in
roughly the same place.

## Why this exists

Asking a model to score a model is easy to do badly. The common failure is a judge that
marks a correct answer down for using vocabulary the rubric happened not to contain, so
the score ends up measuring the rubric's wording rather than the answer. This harness
exists to make that failure visible and testable. Every criterion is graded by one
declared signal, the signal is written down in the report, and a test asserts that a
response which reproduces the reference scores full marks on every criterion.

Two properties are treated as non negotiable:

* A score must be explainable. Every criterion result carries a sentence of reasoning
  naming the signal that produced it.
* A score must be reproducible. The default run is fully offline, uses no clock and no
  random source, and produces identical numbers on every machine.

## Quickstart

```bash
git clone <repository-url>
cd llm-rubric-evaluation-harness
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Inspect the task bank and the rubric library
python -m rubric_harness list-tasks
python -m rubric_harness list-rubrics

# Grade four mock profiles with the offline judge, and write reports/
python -m rubric_harness run --judge

# Grade with only the objective graders, to see which criteria go ungraded
python -m rubric_harness run --out reports_offline

# Calibrate: how much do two judges of different strictness disagree?
python -m rubric_harness agreement

# Run the tests
python -m pytest -q
```

`run` writes three artefacts: `model_scorecard.md` for a reader, `results.csv` for a
spreadsheet, and `run.json` for anything downstream. `report --run reports/run.json`
re-renders the markdown from a saved run without re-calling any provider.

## Project layout

```
llm-rubric-evaluation-harness/
  rubric_harness/
    rubric.py         criteria, level anchors, rubric loading and validation
    tasks.py          the task model, numeric and text answer types
    graders/
      numeric.py      answer_accuracy: extract a number and compare with tolerance
      keyword.py      key_facts: required fact coverage with word form normalisation
      llm_judge.py    holistic: a provider acting as a judge against one criterion
    providers/
      base.py         provider protocol and errors
      mock.py         deterministic mock models and the deterministic mock judge
      registry.py     resolve a provider name, or fail naming the missing key
      openai_provider.py
      anthropic_provider.py
    scoring.py        weighted combination, renormalisation, run evaluation
    runner.py         run the task and provider matrix, with a disk response cache
    agreement.py      percent agreement and Cohen's kappa
    report.py         markdown and JSON rendering
    __main__.py       command line interface
  tasks/
    analytics_tasks.yaml   the task bank: prompts, references, key facts, tolerances
  rubrics/
    quantitative_reasoning.yaml
    statistical_validity.yaml
    key_fact_recall.yaml
    business_document_quality.yaml
  tests/
    test_rubric.py
    test_tasks.py
    test_graders.py
    test_providers.py
```

## The task bank

Six tasks across four domains. Two are numeric, so the answer is a figure with a
tolerance; four are text, so the answer is an explanation graded on its content. Each
task names the rubric that governs it.

| Task | Domain | Answer type | What it asks |
| --- | --- | --- | --- |
| analytics_aggregation_weighting | analytics | numeric | Recompute an overall average from monthly averages and volumes. |
| analytics_churn_denominator | analytics | numeric | Report a monthly churn rate against the correct denominator. |
| stats_multiplicity_decision | experimentation | text | Advise on shipping a result after many comparisons. |
| stats_aggregate_reversal | analytics | text | Explain an aggregate reversal across severity groups. |
| stats_leakage_explanation | machine_learning | text | Explain a leaking feature and say what to do about it. |
| docs_incident_review | business_reporting | text | Write the opening of an executive incident memo. |

The two numeric references, 136.07 and 3.5, are both recomputed by hand from the figures
in the prompt. The key facts on the two text tasks that declare them are checked by the
test suite: every declared fact must be satisfied by its own reference answer, so a task
cannot ship with a fact list that the reference itself fails.

## The rubrics

Four rubrics, fourteen criteria in total, covering thirteen distinct criterion names.
Each criterion carries a weight, a written definition and five level anchors, and names
the grader responsible for it.

| Rubric | Pass threshold | Criteria |
| --- | --- | --- |
| quantitative_reasoning | 0.60 | correctness 0.45, method 0.30, clarity 0.25 |
| statistical_validity | 0.65 | method_appropriateness 0.30, assumption_checking 0.30, interpretation 0.25, transparency 0.15 |
| key_fact_recall | 0.65 | coverage 0.45, reasoning 0.30, clarity 0.25 |
| business_document_quality | 0.60 | accuracy 0.35, structure 0.25, decision_usefulness 0.25, concision 0.15 |

Weights sum to 1.0 in every rubric, and the loader rejects a rubric that does not.

## How grading works

Three graders cover three kinds of criterion.

`answer_accuracy` grades the numeric tasks. It extracts the first number from the
response, handling thousands separators, a currency symbol, a trailing percent sign and
the accounting convention that a parenthesised figure is negative. Within tolerance
scores 1.0, within ten times tolerance scores 0.5, and anything else, including an
unparsable answer, scores 0.0.

`key_facts` grades required fact coverage. A fact counts as covered when every
significant token of the fact appears in the response, after lowercasing, stopword
removal and word form normalisation. The normalisation strips a trailing plural `s`, a
trailing `ing`, a trailing `ed` and a trailing `e`, which is what makes `remove`,
`removes`, `removed` and `removing` one token. It is not a stemmer and does not map
synonyms, so this grader measures recall of a required fact list, not paraphrase.

`holistic` delegates a criterion to a provider acting as a judge, with the criterion
name, its definition, its five anchors, the question, the reference answer and the
response all in the prompt. The reply is parsed for an integer level from 1 to 5. A reply
with no parsable level is recorded as a parse failure and scored at the neutral level 3,
with the raw reply preserved in the result, so a broken judge shows up in the report
rather than quietly inflating or deflating a score. Two criteria graded by the same judge
keep separate scores, keyed `holistic:<criterion>`.

Criterion scores are combined as a weighted mean. A criterion with no applicable grader
is dropped from both the numerator and the denominator, so the remaining weights are
renormalised rather than the missing criterion being scored as zero. This is why the same
run scores differently with and without a judge, and it is a deliberate choice: an
ungraded criterion is unknown, not failed. The report lists every skipped criterion by
name so the renormalisation is never hidden.

## Results

Produced by running the four mock profiles over the whole task bank on the committed
data. The mocks are deterministic, so these figures reproduce exactly.

With the offline judge, every criterion has a grader:

| Model | Mean weighted score | Pass rate |
| --- | --- | --- |
| mock-exact | 1.000 | 100.0 percent |
| mock-off_by_small | 0.472 | 50.0 percent |
| mock-partially_correct | 0.340 | 16.7 percent |
| mock-wrong_method | 0.000 | 0.0 percent |

Without a judge, ten of the fourteen criterion slots have no grader, the remaining
weights are renormalised, and the whole run drops accordingly:

| Model | Mean weighted score | Pass rate |
| --- | --- | --- |
| mock-exact | 0.667 | 66.7 percent |
| mock-off_by_small | 0.345 | 0.0 percent |
| mock-partially_correct | 0.238 | 16.7 percent |
| mock-wrong_method | 0.000 | 0.0 percent |

The mock-exact column is the check that matters. It scores exactly 1.000 on all fourteen
criteria, which is the property the first version of this harness failed: the judge used
to compare a response against the rubric's own wording and marked the correct answer down
to 0.577. The ladder is also strictly monotone at every step, so a model that is wrong in
a smaller way scores higher.

The partially correct column contains the result worth reading. It scores 1.000 on the
incident memo, because that mock returns the full reference on the prompts whose
fingerprint is even, so on that one task it is indistinguishable from the exact profile.
A single mean hides that; the head-to-head table in the report does not.

Agreement between two offline judges reading the same anchors at different strictness, on
all 24 responses:

| Measure | Value |
| --- | --- |
| Items graded by both judges | 24 |
| Percent agreement | 83.3 percent |
| Overall kappa | 0.765 |

Kappa is reported alongside percent agreement because percent agreement flatters a judge
that uses only one level. The two judges agree on 20 of 24 items, and the four
disagreements are all one level apart, which is the behaviour a useful rubric should
produce: a stricter reading of the same anchors moves a borderline response by one level,
not by four.

## The mock profiles

The offline models exist so the harness can be demonstrated, tested and reviewed with no
API key and no network. Each is a pure function of the prompt and the reference table, so
there is no clock, no random source and no hidden state.

| Profile | Behaviour | What it tests |
| --- | --- | --- |
| mock-exact | Returns the reference. | The floor: a correct answer must score 1.000 everywhere. |
| mock-off_by_small | Shifts a bare-figure reference by 2 percent, or drops the closing caveat from a prose reference. | Whether a near miss is scored as a near miss. |
| mock-partially_correct | Returns the reference on half the prompts and a wrong answer on the rest. | Whether a mean hides bimodal behaviour. |
| mock-wrong_method | Replaces a bare figure with one 2.5 times too large, or replaces prose with a confident dismissal. | Whether fluent and wrong is caught. |
| mock-verbose_correct | Wraps the reference in padding. | Whether concision is scored separately from correctness. |

A prose reference is deliberately not number-mangled by the off-by-small profile. An
explanation whose leading figure is silently altered stops being an explanation and
becomes a different error, which would test the mock rather than the grader.

## Plugging in a real model

No model is called by default. Set a key and name the provider, and the same pipeline
grades live responses.

```bash
export OPENAI_API_KEY=...
python -m rubric_harness run --providers openai,openai --judge --judge-provider openai
```

A provider that cannot be constructed fails with a message naming the exact environment
variable to set. Responses are cached on disk under `.cache/responses`, keyed on a hash
of the provider name and the prompt, so a repeated run costs nothing and a re-run after a
grader change does not re-bill the model. `--no-cache` bypasses it.

## Adding a task or a rubric

1. Add the task to `tasks/analytics_tasks.yaml` with its prompt, reference, rubric id and,
   for numeric tasks, a tolerance. A numeric task without a tolerance is rejected at load
   time.
2. Add the rubric to `rubrics/` if a new one is needed. Weights must sum to 1.0, the name
   must match the file name, and every criterion must name a grader.
3. If the task declares key facts, confirm the test suite accepts them: a fact the
   reference itself does not satisfy is a bug in the fact list, not in the grader.
4. Run `python -m pytest -q` and then `python -m rubric_harness run --judge`.

## Limitations

The offline judge is not a model. It scores a single computed signal chosen by the
criterion name, and its numbers are evidence about the mock profiles and about the
plumbing, not about any real model's grading ability. Reported scores from a real judge
should be read with the same suspicion the harness applies to everything else.

The criterion to signal mapping is a lookup on the criterion name. A criterion whose name
matches no known family falls back to plain reference vocabulary overlap, which is a
weaker signal and is treated as such. The families are listed in
`rubric_harness/providers/mock.py` and are deliberately narrow.

The key fact grader matches tokens, not meaning, so a response that states a required
fact in different vocabulary is scored as missing it. That is a recall understatement and
it is the safer direction to err in.

The rubric score is only as good as the anchors, and the anchors in this repository were
written by one person. The agreement figure above shows that two graders of different
strictness disagree on 4 of 24 responses even when both are the same program, which is a
lower bound on how much two human graders would disagree.

The task bank is small, six tasks, and was written to exercise the graders rather than to
be representative of a production evaluation set.
