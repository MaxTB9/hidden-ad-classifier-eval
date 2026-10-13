# Hidden-ad check: a production post classifier vs an expert-labelled reference

This repo lets anyone check one set of numbers: how well a production LLM post classifier spots undisclosed (hidden) advertising, scored against an outside, expert-labelled dataset. It contains the classifier's yes/no output for every post and the script that scores it. It does not contain the classifier itself.

## What was tested and why

The classifier is a Spanish-language social-media post classifier. One of its outputs marks a post as advertising or promotion. The question was whether that output agrees with independent experts, not with labels we wrote ourselves.

The reference is the Gui et al. 2025 Instagram dataset. In it, 3 legal experts labelled 551 posts as hidden ads (127) or organic (424). The design (`design.md`) was declared on 2026-10-05, before the data was read, and the run followed it unchanged.

- Classifier: gemma4:12b via Ollama, temperature 0, seed 42, context 4096, prompt private.
- Positive: the classifier's advertising/promotion output (`flagged_ad = 1` in `predictions.csv`).

## Headline numbers (551 expert-labelled posts)

| Metric | Value |
|---|---|
| Precision | 71.8% (95% 60.5% to 81.0%) |
| Recall | 40.2% (95% 32.0% to 48.9%) |
| F1 | 0.515 (95% bootstrap 0.427 to 0.598) |
| Accuracy | 82.6% (always "organic": 77.0%) |

Confusion: TP 51, FP 20, FN 76, TN 404. For comparison, always answering "ad" gives an F1 of 0.375. On the 592 openly disclosed ads, scored separately, recall is 64.4% (381 of 592). Full table: `results.md`.

**Honest reading:** when the classifier flags a post as advertising, it is right about 7 times in 10. But it undercounts hidden ads by more than half: it misses 76 of the 127. It is reliable when it fires; it is not a complete ad detector.

## Reproduce (Python 3.10+)

```bash
pip install -r requirements.txt
curl -L --create-dirs -o data/Dataset.csv https://raw.githubusercontent.com/HaoyangGui/Evaluating-LLM-Generated-Legal-Explanations/main/dataset/Dataset.csv
python score.py --dataset data/Dataset.csv --predictions predictions.csv
```

`score.py` checks the dataset's md5 (`aae56369561b23a0be5864159264eee5`) and warns if yours differs. Add `--json out.json` to save the numbers.

## What is in the repo and what is not

In the repo:
- `predictions.csv`: `row_id` (the dataset's `id`) and `flagged_ad` (1/0) for all 1,143 posts.
- `score.py`: scores the predictions.
- `results.md`: the numbers above, in full.
- `design.md`: the pre-registered design.

Not in the repo, by design: the prompt, the pipeline, and any model configuration beyond the line above. The authors' data is not included either: no captions, no labels, no categories.

## Dataset credit

Gui et al. 2025, "Evaluating LLM-Generated Legal Explanations for Regulatory Compliance in Social Media Influencer Marketing", https://github.com/HaoyangGui/Evaluating-LLM-Generated-Legal-Explanations.

That repository states no licence, so the data is not redistributed here. Download it from the authors (command above). All credit for the labels goes to them.

## Limitations

- **Recall is 40%.** Most hidden ads go unflagged.
- **Not an ad detector by design.** The positive class is one output of a broader post classifier, not a purpose-built ad detector.
- **Language mismatch.** The prompt was written for Spanish posts and was run unchanged on this English set.
- **Text used.** Every post was scored on the authors' `caption_cleaned` text.
- **Disclosed set.** The disclosed set is openly disclosed ads, scored on the authors' caption_cleaned text. 147 of the 592 had a marker such as #ad removed by the authors; the other 445 had no marker in the caption.
- **Selection unknown.** The authors do not state how the 592 disclosed ads were selected.
- **Truncation.** 7 posts were truncated at 1,500 characters, as in production.
- **One run.** This is a single run, at temperature 0, on one dataset.

## Author

Maximiliano Biatturi Anton (GitHub: MaxTB9). Licence: MIT for the code and predictions in this repo (`LICENSE`); the dataset is not covered.
