import os
import io
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vehicle_failure_artifacts.joblib")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Vehicle Failure Prediction", page_icon="🛠️", layout="centered")
st.title("🛠️ Vehicle Failure Prediction (Predictive Maintenance)")
st.write("Predict **failure risk** and   **failure type**, with **Explainable AI** and a **threshold performance dashboard**.")

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

artifacts = load_artifacts()

rf_binary = artifacts["rf_binary_model"]
rf_type = artifacts["rf_failure_type_model"]
feature_cols = artifacts["feature_cols"]
FINAL_THRESHOLD = float(artifacts["final_threshold"])

# XAI
feature_importances = artifacts.get("feature_importances", None)
train_stats = artifacts.get("train_stats", None)

# Performance arrays (for slider + confusion matrix)
y_test_arr = artifacts.get("y_test", None)
y_prob_test_arr = artifacts.get("y_prob_test", None)


# ----------------------------
# Helpers
# ----------------------------
def make_input_row(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min):
    machine_type = str(machine_type).strip().upper()
    if machine_type not in {"L", "M", "H"}:
        raise ValueError("machine_type must be one of: L, M, H")

    row = {
        "Air temperature [K]": float(air_k),
        "Process temperature [K]": float(process_k),
        "Rotational speed [rpm]": int(rpm),
        "Torque [Nm]": float(torque_nm),
        "Tool wear [min]": int(tool_wear_min),
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0,
    }

    X_one = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
    return X_one


def predict_all(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min):
    X_one = make_input_row(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min)

    prob_fail = float(rf_binary.predict_proba(X_one)[0, 1])
    will_fail = prob_fail >= FINAL_THRESHOLD

    failure_type = None
    if will_fail:
        failure_type = str(rf_type.predict(X_one)[0])

    return X_one, {
        "will_fail": will_fail,
        "prob_fail": prob_fail,
        "threshold": FINAL_THRESHOLD,
        "failure_type": failure_type
    }


def explain_input_vs_training(X_one: pd.DataFrame, stats: pd.DataFrame, top_k: int = 5):
    if stats is None:
        return ["Training statistics not found. Re-save artifacts with `train_stats`."]

    row = X_one.iloc[0]
    diffs = (row - stats["mean"]).abs().sort_values(ascending=False)

    lines = []
    for feat in diffs.head(top_k).index:
        val = float(row[feat])
        mean = float(stats.loc[feat, "mean"])
        mn = float(stats.loc[feat, "min"])
        mx = float(stats.loc[feat, "max"])
        direction = "higher than" if val > mean else "lower than"
        lines.append(
            f"- **{feat}** is {direction} typical (value={val:.2f}, mean={mean:.2f}, range=[{mn:.2f}, {mx:.2f}])"
        )
    return lines


def risk_label(prob, threshold):
    if prob < 0.20:
        return "Low"
    elif prob < threshold:
        return "Medium"
    elif prob < 0.60:
        return "High"
    return "Critical"


def recommendation(prob, threshold):
    r = risk_label(prob, threshold)
    if r == "Low":
        return "✅ Monitor normally."
    if r == "Medium":
        return "🟡 Inspect soon (schedule check)."
    if r == "High":
        return "🟠 Schedule preventive maintenance."
    return "🔴 Stop/inspect immediately (critical risk)."


def plot_confusion_matrix(cm, class_names=("No Failure", "Failure")):
    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Write numbers in cells (fixed)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    return fig



def build_pdf_report(pred_row: dict) -> bytes:
    """
    Create a simple one-page PDF report from the latest prediction.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Vehicle Failure Prediction Report")
    y -= 30

    c.setFont("Helvetica", 11)
    for k, v in pred_row.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# ----------------------------
# Tabs
# ----------------------------
tab_predict, tab_xai, tab_perf, tab_downloads, tab_about = st.tabs(
    ["Predict", "Explainability (XAI)", "Performance Dashboard", "Downloads", "About"]
)

# ----------------------------
# Predict tab
# ----------------------------
with tab_predict:
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=0)
        air_k = st.number_input("Air temperature [K]", min_value=200.0, max_value=400.0, value=300.0, step=0.1)
        process_k = st.number_input("Process temperature [K]", min_value=200.0, max_value=450.0, value=310.0, step=0.1)

    with col2:
        rpm = st.number_input("Rotational speed [rpm]", min_value=0, max_value=5000, value=1500, step=1)
        torque_nm = st.number_input("Torque [Nm]", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        tool_wear_min = st.number_input("Tool wear [min]", min_value=0, max_value=500, value=200, step=1)

    st.divider()

    if st.button("Predict"):
        X_one, out = predict_all(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min)

        st.subheader("Result")
        st.write(f"**Failure probability:** `{out['prob_fail']:.3f}`")
        st.write(f"**Threshold used:** `{out['threshold']:.2f}`")
        st.write(f"**Risk level:** `{risk_label(out['prob_fail'], out['threshold'])}`")
        st.info(recommendation(out['prob_fail'], out['threshold']))

        if out["will_fail"]:
            st.error("⚠️ Prediction: FAILURE likely")
            st.write(f"**Predicted failure type:** `{out['failure_type']}`")
        else:
            st.success("✅ Prediction: NO FAILURE likely")
            st.write("**Predicted failure type:** `None`")

        with st.expander("Show model input row"):
            st.dataframe(X_one)

        # Save latest prediction for other tabs
        st.session_state["last_X_one"] = X_one
        st.session_state["last_out"] = out
        st.session_state["last_inputs"] = {
            "Type": machine_type,
            "Air temperature [K]": air_k,
            "Process temperature [K]": process_k,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque_nm,
            "Tool wear [min]": tool_wear_min,
        }

# ----------------------------
# XAI tab
# ----------------------------
with tab_xai:
    st.subheader("Explainability (XAI)")

    if feature_importances is None:
        st.warning("Feature importances not found. Re-save artifacts with `feature_importances`.")
    else:
        st.write("### Global: Top feature importances")
        st.bar_chart(feature_importances.head(10))

    st.write("### Local: Why this prediction?")
    if "last_X_one" not in st.session_state:
        st.info("Make a prediction in **Predict** first.")
    else:
        lines = explain_input_vs_training(st.session_state["last_X_one"], train_stats, top_k=5)
        st.markdown("\n".join(lines))

# ----------------------------
# Performance Dashboard tab
# ----------------------------
with tab_perf:
    st.subheader("Threshold Slider (Recall vs False Alarms)")

    if y_test_arr is None or y_prob_test_arr is None:
        st.warning(
            "Performance arrays not found. In the notebook, save `y_test` and `y_prob_test` into the joblib artifacts."
        )
    else:
        t = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=float(FINAL_THRESHOLD), step=0.01)

        y_pred_t = (pd.Series(y_prob_test_arr) >= t).astype(int).values
        cm = confusion_matrix(y_test_arr, y_pred_t)

        tn, fp, fn, tp = cm.ravel()
        recall_fail = tp / (tp + fn) if (tp + fn) else 0.0
        precision_fail = tp / (tp + fp) if (tp + fp) else 0.0

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Failure Recall", f"{recall_fail:.3f}")
        colB.metric("Failure Precision", f"{precision_fail:.3f}")
        colC.metric("Missed Failures (FN)", f"{fn}")
        colD.metric("False Alarms (FP)", f"{fp}")

        st.pyplot(plot_confusion_matrix(cm))

        with st.expander("Classification report at this threshold"):
            st.text(classification_report(y_test_arr, y_pred_t, zero_division=0))

# ----------------------------
# Downloads tab
# ----------------------------
with tab_downloads:
    st.subheader("Download Prediction Report")

    if "last_out" not in st.session_state:
        st.info("Make a prediction in **Predict** first, then come back here to download the report.")
    else:
        out = st.session_state["last_out"]
        inputs = st.session_state.get("last_inputs", {})

        report_row = {
            **inputs,
            "Failure probability": round(out["prob_fail"], 4),
            "Threshold used": round(out["threshold"], 2),
            "Will fail?": "Yes" if out["will_fail"] else "No",
            "Failure type": out["failure_type"] if out["failure_type"] else "None",
            "Risk level": risk_label(out["prob_fail"], out["threshold"]),
            "Recommendation": recommendation(out["prob_fail"], out["threshold"]),
        }

        df_report = pd.DataFrame([report_row])

        # CSV download
        csv_bytes = df_report.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="prediction_report.csv",
            mime="text/csv"
        )

        # PDF download
        pdf_bytes = build_pdf_report(report_row)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )

        st.write("Preview:")
        st.dataframe(df_report)

# ----------------------------
# About tab
# ----------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
**Pipeline**
1. Binary Random Forest predicts failure probability  
2. Threshold `{FINAL_THRESHOLD:.2f}` converts probability → Fail/No Fail  
3. If Fail → multiclass model predicts failure type  
4. XAI shows global feature importance + local explanation

**Why threshold tuning?**  
Predictive maintenance prioritizes catching failures (high recall) over minimizing false alarms.
        """
    )
