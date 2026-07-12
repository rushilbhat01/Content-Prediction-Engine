import pathlib
src = pathlib.Path("model_train.py").read_text(encoding="utf-8")
pathlib.Path("model_train_food.py").write_text(
    src.replace('NICHE      = "fitness"', 'NICHE      = "food"'), encoding="utf-8"
)
pathlib.Path("model_train_fitness.py").write_text(src, encoding="utf-8")
print("Done")
