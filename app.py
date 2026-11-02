import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Marks Predictor & Weak Area Detector",
    page_icon="🎓",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    root = Path.cwd()
    artifacts_dir = root / "artifacts"
    
    reg_model = joblib.load(artifacts_dir / "best_regression_model.joblib")
    clf_model = joblib.load(artifacts_dir / "best_classification_model.joblib")
    importance_df = joblib.load(artifacts_dir / "regression_feature_importance.joblib")
    return reg_model, clf_model, importance_df

try:
    reg_model, clf_model, importance_df = load_artifacts()
except Exception as e:
    st.error(f"Models not found in artifacts/ folder. Please run your Jupyter notebook or training script first! Error: {e}")
    st.stop()

st.title("🎓 Student Marks Predictor & Weak Area Detector")
st.markdown("Predict your final marks (0-100), pass/fail risk level, and discover weak areas with actionable improvement recommendations.")

# Sidebar input controls for student
st.sidebar.header("Student Input Panel")

study_hours = st.sidebar.slider("Study Hours per Day", 0.5, 8.0, 3.0, 0.5)
attendance_pct = st.sidebar.slider("Attendance Percentage (%)", 40.0, 100.0, 85.0, 1.0)
previous_score = st.sidebar.slider("Previous Test Score (out of 100)", 0.0, 100.0, 65.0, 1.0)
subject = st.sidebar.selectbox("Subject", ["math", "portuguese"])
assignment_submitted = st.sidebar.selectbox("Assignments Submitted on Time?", ["Yes", "Mostly", "No"])
sleep_hours = st.sidebar.slider("Average Sleep Hours per Night", 4.0, 10.0, 7.0, 0.5)
internet_usage = st.sidebar.selectbox("Home Internet Access / Usage", ["Yes", "No", "Medium"])

# Mapping simple inputs to UCI features
def map_study_hours(hours):
    weekly = hours * 7
    if weekly < 2: return 1
    elif weekly <= 5: return 2
    elif weekly <= 10: return 3
    else: return 4

def map_attendance_to_absences(att_pct):
    att_pct = float(np.clip(att_pct, 0, 100))
    return int(round((100 - att_pct) / 100 * 30))

def map_score_to_grade20(score):
    return int(round(float(np.clip(score, 0, 100)) / 5))

# Build sample row matching training features
# We load default template from a dummy frame or hardcode medians from our dataset summary
sample_row = {
    'school': 'GP',
    'sex': 'F',
    'age': 17.0,
    'address': 'U',
    'famsize': 'GT3',
    'Pstatus': 'T',
    'Medu': 3.0,
    'Fedu': 3.0,
    'Mjob': 'other',
    'Fjob': 'other',
    'reason': 'course',
    'guardian': 'mother',
    'traveltime': 1.0,
    'studytime': map_study_hours(study_hours),
    'failures': 0.0,
    'schoolsup': 'no',
    'famsup': 'yes',
    'paid': 'no',
    'activities': 'yes',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': 'no' if internet_usage == 'No' else 'yes',
    'romantic': 'no',
    'famrel': 4.0,
    'freetime': 3.0,
    'goout': 3.0,
    'Dalc': 1.0,
    'Walc': 2.0,
    'health': 4.0,
    'absences': map_attendance_to_absences(attendance_pct),
    'G1': map_score_to_grade20(previous_score),
    'G2': map_score_to_grade20(previous_score),
    'subject': subject,
    'attendance_proxy_pct': attendance_pct
}

input_df = pd.DataFrame([sample_row])

if st.sidebar.button("Predict & Analyze", type="primary"):
    # Predictions
    pred_g3_20 = reg_model.predict(input_df)[0]
    pred_g3_100 = float(np.clip(pred_g3_20 * 5, 0, 100))
    pred_pass_prob = clf_model.predict_proba(input_df)[0][1]
    pass_fail_pred = clf_model.predict(input_df)[0]
    
    # Risk Level
    if pred_g3_100 < 50:
        risk = "High Risk 🔴"
        risk_color = "red"
    elif pred_g3_100 < 70:
        risk = "Moderate Risk 🟡"
        risk_color = "orange"
    else:
        risk = "Low Risk / Safe 🟢"
        risk_color = "green"
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Final Marks (0-100)", f"{pred_g3_100:.1f} / 100")
    with col2:
        status_text = "Pass" if pass_fail_pred == 1 else "Fail Risk"
        st.metric("Predicted Status", status_text, f"{pred_pass_prob*100:.1f}% confidence")
    with col3:
        st.markdown(f"**Risk Level:**\n\n### {risk}")
        
    st.markdown("---")
    st.subheader("🔍 Weak Area Detector & Actionable Feedback")
    
    # Feature weakness logic
    importance_lookup = dict(zip(importance_df['feature'], np.maximum(importance_df['importance_mean'], 0)))
    
    actionable_checks = [
        ("attendance_proxy_pct", attendance_pct < 75, "Low attendance / too many absences"),
        ("studytime", sample_row['studytime'] <= 1, "Low weekly study time"),
        ("G1", sample_row['G1'] < 10, "Low previous test performance"),
        ("failures", sample_row['failures'] > 0, "Past academic failure history"),
        ("internet", sample_row['internet'] == 'no', "Lack of home internet / digital resources")
    ]
    
    detected_weaknesses = []
    for feat, is_weak, reason in actionable_checks:
        if is_weak:
            imp = importance_lookup.get(feat, 0.01)
            detected_weaknesses.append((reason, imp))
            
    detected_weaknesses = sorted(detected_weaknesses, key=lambda x: x[1], reverse=True)
    
    if detected_weaknesses:
        for reason, _ in detected_weaknesses:
            st.warning(f"⚠️ **Weak Area Detected:** {reason}")
            
        st.markdown("### 💡 Recommended Improvements:")
        recommendations = {
            "Low attendance / too many absences": "Try to maintain at least 80%+ attendance and review notes for missed classes.",
            "Low weekly study time": "Increase daily focused study blocks by 45-60 minutes.",
            "Low previous test performance": "Focus extra revision on foundational concepts from your earlier term tests.",
            "Past academic failure history": "Schedule weekly doubt-clearing sessions with your mentors or peers.",
            "Lack of home internet / digital resources": "Download offline study material and practice sets in advance."
        }
        for reason, _ in detected_weaknesses:
            if reason in recommendations:
                st.info(f"✔️ {recommendations[reason]}")
    else:
        st.success("🌟 Great job! No major weak areas detected based on your current inputs.")

else:
    st.info("👈 Sidebar me apne details enter karein aur **Predict & Analyze** button par click karein.")