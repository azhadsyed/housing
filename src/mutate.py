# poetry run python src/mutate.py "C:\github\housing_local\pickle\201124_2315_results_newyork_apa_todayFalse.pkl" "C:\storage\test.csv"

import pandas as pd

import sys

src = sys.argv[1]
dst = sys.argv[2]

data = pd.read_pickle(src)
data.to_csv(dst)