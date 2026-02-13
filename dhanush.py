# ======================================================
# FEATURE ENGINEERING MODEL (ADVANCED)
# ======================================================

print("\nStarting Feature Engineering...")

# 1. Delivery Delay Feature
if "delivery_time" in df.columns and "expected_time" in df.columns:
    df["delivery_delay"] = df["delivery_time"] - df["expected_time"]

# 2. Delivery Efficiency Score
if "ontime_rate" in df.columns and "delivery_time" in df.columns:
    df["efficiency_score"] = df["ontime_rate"] / (df["delivery_time"] + 1)

# 3. Speed Feature
if "distance" in df.columns and "delivery_time" in df.columns:
    df["delivery_speed"] = df["distance"] / (df["delivery_time"] + 1)

# 4. Risk Category Feature
if "ontime_rate" in df.columns:
    df["risk_category"] = pd.cut(
        df["ontime_rate"],
        bins=[0, 0.7, 0.9, 1],
        labels=[2, 1, 0]
    )

# 5. Performance Consistency Feature
if "ontime_rate" in df.columns:
    df["performance_variation"] = abs(df["ontime_rate"] - df["ontime_rate"].mean())

# 6. Outlier Flag Feature
if "delivery_delay" in df.columns:
    threshold = df["delivery_delay"].quantile(0.95)
    df["extreme_delay_flag"] = (df["delivery_delay"] > threshold).astype(int)

# 7. Composite Risk Score
if "delivery_speed" in df.columns and "efficiency_score" in df.columns:
    df["composite_risk_score"] = (
        df["delivery_speed"] * df["efficiency_score"]
    )

print("Feature Engineering Completed Successfully.")
print("New Dataset Shape:", df.shape)
