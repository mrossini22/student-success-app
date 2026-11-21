import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓")

FEATURES = {
    "Submission.Actions": "Submission actions",
    "Quiz.Actions": "Quiz actions",
    "Forum.Actions": "Forum actions",
    "Create.Actions": "Create actions",
    "Read.Actions": "Read actions",
    "Update.Actions": "Update actions",
    "Delete.Actions": "Delete actions",
    "Days.Logged.In": "Days logged in",
}


@st.cache_resource
def train():
    df = pd.read_csv("data.csv").drop_duplicates()  # augmented data has exact duplicates
    X = df[list(FEATURES)]
    y = (df["rating"] == "High").astype(int)  # High -> Pass, Low -> Fail
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(Xtr, ytr)
    pred = model.predict(Xte)
    report = classification_report(yte, pred, target_names=["Fail", "Pass"], output_dict=True)
    # refit on all data for the deployed model
    model.fit(X, y)
    return model, df, accuracy_score(yte, pred), report


model, df, acc, report = train()

st.title("🎓 Student Success Predictor")
st.write("Enter a student's learning-activity record to predict whether they are likely to **pass or fail**.")

with st.form("student"):
    c1, c2 = st.columns(2)
    vals = {}
    for i, (col, label) in enumerate(FEATURES.items()):
        mx = int(df[col].max())
        with (c1 if i % 2 == 0 else c2):
            vals[col] = st.number_input(
                label, min_value=0, value=int(df[col].median()), step=1,
                help=f"Typical (median): {int(df[col].median())}; max in training data: {mx}",
            )
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    x = pd.DataFrame([vals])[list(FEATURES)]
    p_pass = float(model.predict_proba(x)[0][1])
    if p_pass >= 0.5:
        st.success(f"✅ Likely to PASS (confidence {p_pass:.0%})")
    else:
        st.error(f"⚠️ At risk of FAILING (confidence {1 - p_pass:.0%})")
    st.progress(p_pass, text=f"Probability of passing: {p_pass:.0%}")
    imp = pd.Series(model.feature_importances_, index=list(FEATURES.values())).sort_values()
    st.caption("What the model relies on overall (feature importance)")
    st.bar_chart(imp, horizontal=True)

with st.expander("About the model"):
    st.write(
        f"Random Forest trained on {len(df):,} unique student records "
        f"(duplicates removed). Hold-out accuracy: **{acc:.1%}**."
    )
    st.dataframe(pd.DataFrame(report).T.round(3).loc[["Fail", "Pass"]])
    st.caption(
        "Dataset ratings are mapped as High → Pass and Low → Fail. "
        "This is a decision-support tool based on online activity only; "
        "it should not be the sole basis for decisions about a student."
    )
