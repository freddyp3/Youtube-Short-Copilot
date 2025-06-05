# feedback loop thing

idea: gemini makes a thumbnail (or picks the best frame out of the video),
then a xgboost model scores it on predicted engagement. if the score is
too low it goes back and tries again.

- `frames.py` pulls candidate frames out of a video
- `thumb_gen.py` = gemini picks a frame
- `features.py` + `train_thumb_model.py` = the xgboost judge
- `thumb_loop.py` = generate -> score -> retry

## status

model kind of works on like 8 thumbnails. needs way more data before the
acceptance score means anything.

accept score is set way too low rn, its basically accepting anything.
scores all cluster near zero on the tiny training set.
