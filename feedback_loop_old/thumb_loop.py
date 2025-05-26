import xgboost as xgb
from thumb_gen import pick_frame, gen_thumbnail
from features import get_features

model = xgb.XGBRegressor()
model.load_model("thumb_model.json")

ACCEPT_SCORE = 0.05


def score(path):
    return model.predict([get_features(path)])[0]


def make_thumbnail(video_path, tries=5):
    """keep going until the model thinks the thumbnail is good enough"""
    best_score = 0
    for i in range(tries):
        cand = gen_thumbnail(out="cand_%d.png" % i)
        if cand is None:
            cand = pick_frame(video_path)
        s = score(cand)
        print("try", i + 1, "->", cand, "score", s)
        if s > best_score:
            best_score = s
            return cand
    return None


if __name__ == "__main__":
    make_thumbnail("test.mp4")
