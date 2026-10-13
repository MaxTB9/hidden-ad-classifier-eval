# Results

Classifier: gemma4:12b via Ollama, temperature 0, seed 42, context 4096, prompt private. Positive = the classifier's advertising/promotion output (`flagged_ad = 1`). Run on 2026-10-05; design declared the same day before the data was read (see `design.md`).

Reference data: Gui et al. 2025, `dataset/Dataset.csv` from https://github.com/HaoyangGui/Evaluating-LLM-Generated-Legal-Explanations (1,143 rows). md5 `aae56369561b23a0be5864159264eee5`; `score.py` warns if yours differs.

All 1,143 posts were scored; no call failed to parse and none needed a retry.

## Main test: 551 expert-labelled posts (127 hidden ads, 424 organic)

| Metric | Value |
|---|---|
| Precision | 71.8% (95% Wilson 60.5% to 81.0%) |
| Recall | 40.2% (95% Wilson 32.0% to 48.9%) |
| F1 | 0.515 (95% bootstrap 0.427 to 0.598; 2,000 draws, seed 51) |
| Accuracy | 82.6% (95% Wilson 79.2% to 85.5%) |
| Context: F1 of always answering "ad" | 0.375 |
| Context: accuracy of always answering "organic" | 77.0% |

Confusion:

| | Flagged ad | Not flagged |
|---|---|---|
| Hidden ad (127) | TP 51 | FN 76 |
| Organic (424) | FP 20 | TN 404 |

## Secondary: 592 openly disclosed ads (kept apart, never mixed into the main test)

Recall: 64.4% (95% Wilson 60.4% to 68.1%), 381 of 592.

These are openly disclosed ads, scored on the authors' caption_cleaned text (147 of the 592 had a marker such as #ad removed by the authors; the other 445 had no marker in the caption).

## Notes on the text sent

- Text = the authors' `caption_cleaned` column. On the 551 it differs from `caption` only by whitespace, except 7 hidden ads whose trailing "ad" the authors removed and 2 organic posts that lost a hashtag starting with "ad". Using `caption` would have handed the classifier the answer on those 7.
- 7 posts exceed 1,500 characters (4 in the 551, 3 in the 592) and were truncated at 1,500 characters, as in production.
