import pandas as pd

R1 = "outputs/predictions/round1_200nm_consensus.csv"
R0 = "outputs/predictions/round0_200nm_consensus.csv"

df = pd.read_csv(R1)
print("=== Round 1 label distribution ===")
print(df["final_classification"].value_counts())
print()
print("=== Agreement / confidence ===")
print(df[["label_agreement", "mean_majority_confidence",
          "calibrated_confidence", "review_flag"]].describe().round(3))
print()
total_cost = df["running_cost_usd"].max()
print(f"=== Cost ===\nTotal cost: ${total_cost:.2f}")
print()

acc = df[(df["final_classification"].isin(["HEALTHY", "UNHEALTHY"]))
         & (df["label_agreement"] >= 0.67)
         & (df["calibrated_confidence"] >= 0.55)]
print("=== Accept sim (agreement>=0.67 & cal_conf>=0.55) ===")
print(acc["final_classification"].value_counts())
print(f"Total accepted: {len(acc)} / {len(df)}")
print()

r0 = pd.read_csv(R0)
both = pd.concat([r0, df], ignore_index=True)
acc_both = both[(both["final_classification"].isin(["HEALTHY", "UNHEALTHY"]))
                & (both["label_agreement"] >= 0.67)
                & (both["calibrated_confidence"] >= 0.55)]
print("=== Combined R0+R1 accepted ===")
print(acc_both["final_classification"].value_counts())
print(f"Combined accepted: {len(acc_both)} / {len(both)}")
