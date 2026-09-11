"""Row-level keys: TE on multi-column combinations (tests for row-level generator collapse)."""
from src.cv import run
from src.features.static import GROUPS
from src.features.target_enc import TESpec
INC, COM = "Annual_Income_USD", "Daily_Commute_km"
full = GROUPS["raw"] + GROUPS["inc_stats"] + GROUPS["com_stats"] + GROUPS["digits"]
def temulti(m=3):
    return [TESpec((INC,), m=m), TESpec((COM,), m=m), TESpec((INC,), m=m, binwidth=100), TESpec((INC,), m=m, binwidth=500),
            TESpec((INC,), m=m, binwidth=2000), TESpec((COM,), m=m, binwidth=1), TESpec((COM,), m=m, binwidth=5)]
LOW = ("Age", "Gender", "City_Type", "Number_of_Cars_Owned", "Current_Car_Type", "Charging_Stations_Near_Home",
       "Charging_Stations_Near_Work", "Home_Charging_Possible", "Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level")
K8 = ("Age", "City_Type", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level",
      "Subsidy_Available", "Range_Anxiety_Level", "Home_Charging_Possible")
REF = "lgb_full_temulti_m3"
run("lgb_m3_rowkeys", full, te_specs=temulti(3) + [TESpec(LOW, m=10, decimals=0), TESpec(K8, m=10, decimals=0),
    TESpec(LOW + (COM,), m=10, decimals=1)], static_version="v2", ref=REF)
