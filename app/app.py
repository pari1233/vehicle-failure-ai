import streamlit as st
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vehicle_failure_artifacts.joblib")

st.set_page_config(page_title="Vehicle Failure Prediction", page_icon="🛠️")

st.title("🛠️ Vehicle Failure Prediction (Predictive Maintenance)")
st.write("Enter sensor values and machine type to predict failure risk and failure type.")

# --- Load saved artifacts ---
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load(MODEL_PATH)
    return artifacts

artifacts = load_artifacts()
rf_binary = artifacts["rf_binary_model"]
rf_type = artifacts["rf_failure_type_model"]
feature_cols = artifacts["feature_cols"]
FINAL_THRESHOLD = artifacts["final_threshold"]


def make_input_row(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min):
    machine_type = str(machine_type).strip().upper()
    if machine_type not in ["L", "M", "H"]:
        raise ValueError("machine_type must be one of: L, M, H")

    row = {
        "Air temperature [K]": float(air_k),
        "Process temperature [K]": float(process_k),
        "Rotational speed [rpm]": int(rpm),
        "Torque [Nm]": float(torque_nm),
        "Tool wear [min]": int(tool_wear_min),
        # Encoding for your dataset: Type_L, Type_M (Type_H is when both 0)
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0,
    }

    X_one = pd.DataFrame([row])
    X_one = X_one.reindex(columns=feature_cols, fill_value=0)
    return X_one


def predict(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min):
    X_one = make_input_row(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min)
    prob_fail = rf_binary.predict_proba(X_one)[0, 1]
    will_fail = prob_fail >= FINAL_THRESHOLD

    result = {
        "will_fail": will_fail,
        "prob_fail": float(prob_fail),
        "threshold": float(FINAL_THRESHOLD),
        "failure_type": None
    }

    if will_fail:
        result["failure_type"] = str(rf_type.predict(X_one)[0])

    return result


# --- UI Inputs ---
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
    try:
        out = predict(machine_type, air_k, process_k, rpm, torque_nm, tool_wear_min)

        st.subheader("Result")
        st.write(f"**Failure probability:** {out['prob_fail']:.3f}")
        st.write(f"**Threshold used:** {out['threshold']:.2f}")

        if out["will_fail"]:
            st.error("⚠️ Prediction: FAILURE likely")
            st.write(f"**Predicted failure type:** {out['failure_type']}")
        else:
            st.success("✅ Prediction: NO FAILURE likely")
            st.write("**Predicted failure type:** None")

        # Optional: show raw inputs
        with st.expander("Show input values"):
            st.json({
                "Type": machine_type,
                "Air temperature [K]": air_k,
                "Process temperature [K]": process_k,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque_nm,
                "Tool wear [min]": tool_wear_min
            })

    except Exception as e:
        st.exception(e)