import pathlib

src = pathlib.Path("model_train.py").read_text(encoding="utf-8")
fixed = src.replace("\u2192", "->").replace("\u2190", "<-")
pathlib.Path("model_train.py").write_text(fixed, encoding="utf-8")
pathlib.Path("model_train_fitness.py").write_text(fixed, encoding="utf-8")
food = fixed.replace('NICHE      = "fitness"', 'NICHE      = "food"')
pathlib.Path("model_train_food.py").write_text(food, encoding="utf-8")
print("Done - fixed arrows and regenerated both scripts")
