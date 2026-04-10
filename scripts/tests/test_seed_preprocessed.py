# test_seed_preprocessed.py
from datasets.seed_preprocessed import extract_seed_feature

if __name__ == "__main__":
    trainx, trainy, testx, testy = extract_seed_feature()
    print("Train X:", trainx.shape)
    print("Train y:", trainy.shape)
    print("Test  X:", testx.shape)
    print("Test  y:", testy.shape)
    print("trainy unique:", set(trainy.tolist()))
    print("testy  unique:", set(testy.tolist()))
