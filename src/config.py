from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "proc"
EXP = ROOT / "experiments"
SUB = ROOT / "submissions"
for _p in (RAW, PROC, EXP, SUB):
    _p.mkdir(parents=True, exist_ok=True)

COMP = "playground-series-s6e9"
ORIG_DATASET = "itzzomkar/ev-adoption-behavior-and-range-anxiety"
ORIG_FILE = "EV_Adoption_and_Range_Anxiety_Dataset.csv"

ID = "id"
TARGET = "Will_Buy_EV"
N_FOLDS = 10
FOLD_SEED = 0

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]
NUM_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
FEATURES = NUM_COLS + CAT_COLS

# fixed integer codings so every model/feature sees the same ids
CAT_MAPS = {
    "Gender": {"Male": 0, "Female": 1, "Other": 2},
    "City_Type": {"Rural": 0, "Suburban": 1, "Urban": 2},
    "Current_Car_Type": {"Hatchback": 0, "Sedan": 1, "SUV": 2, "Truck": 3},
    "Home_Charging_Possible": {"No": 0, "Yes": 1},
    "Subsidy_Available": {"No": 0, "Yes": 1},
    "Range_Anxiety_Level": {"Low": 0, "Medium": 1, "High": 2},
}
