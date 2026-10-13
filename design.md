# Pre-registered design (declared 2026-10-05, before any data was read)

Written down the same day, before the dataset was opened; the run followed it unchanged.

- **Reference set:** Gui et al. 2025, `dataset/Dataset.csv` from https://github.com/HaoyangGui/Evaluating-LLM-Generated-Legal-Explanations.
- **Main test:** the 551 expert-labelled posts only: 127 hidden (undisclosed) ads vs 424 organic posts, labelled by 3 legal experts (agreement alpha 0.74, as reported by the authors).
- **Classifier:** gemma4:12b with the production prompt, unchanged, temperature 0, the production seed.
- **Positive:** the classifier gives its advertising/promotion output. Everything else counts as negative, and so does a call that fails to parse.
- **Metrics (main test):**
  - precision and recall for "ad", with Wilson 95% intervals;
  - F1, with a 95% percentile interval from a bootstrap over the 551 posts (2,000 draws, seed 51);
  - accuracy;
  - the binary confusion counts.
- **Secondary check:** the 592 disclosed ads are reported separately, as recall only. They are never mixed into the main test.
- **Context, not declared metrics:** the F1 of always answering "ad" and the accuracy of always answering "organic".
- **Use of the data:** the dataset states no licence, so the design declared it for internal testing only. The decision to publish the classifier's own outputs and aggregate scores, never the authors' text or labels, was taken on 2026-10-10, after the run.
- **Not published here:** the design also declared the confusion of the hidden ads against the classifier's other output classes. Those classes belong to the private prompt, so this repo gives the binary confusion counts only.

