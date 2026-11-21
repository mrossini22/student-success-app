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

st.markdown(
    """
    <style>
    :root {color-scheme: light;}
    .stApp, [data-testid="stHeader"], [data-testid="stAppViewContainer"] {background: #fff !important;}
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp caption,
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {color: #000 !important;}
    .block-container {max-width: 760px; padding-top: 2rem;}
    h1, h2, h3 {letter-spacing: -0.5px;}
    div[data-testid="stForm"] {border: 1px solid #000; border-radius: 12px; background: #fff;}
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] input {background: #fff !important; color: #000 !important;}
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {border: 1px solid #000 !important;}
    div[data-testid="stNumberInput"] button {background: #fff !important; color: #000 !important;}
    div[data-testid="stFormSubmitButton"] button {
        background: #000 !important; border: 1px solid #000; border-radius: 10px; font-weight: 600;}
    div[data-testid="stFormSubmitButton"] button, div[data-testid="stFormSubmitButton"] button * {color: #fff !important;}
    div[data-testid="stFormSubmitButton"] button:hover {background: #333 !important;}
    button[data-baseweb="tab"] {font-weight: 600;}
    table.perf {width: 100%; border-collapse: collapse; margin: 0.5rem 0 1rem 0;}
    table.perf th, table.perf td {border: 1px solid #000; padding: 0.6rem 0.9rem; text-align: right; background: #fff;}
    table.perf thead th {background: #000; color: #fff !important; text-align: right;}
    table.perf thead th, table.perf thead th * {color: #fff !important;}
    table.perf tbody th {text-align: left; font-weight: 700;}
    table.perf th:first-child {text-align: left;}
    [data-baseweb="tab-highlight"] {background: #000 !important;}
    div[data-testid="stFormSubmitButton"] button:focus {box-shadow: none !important; border-color: #000 !important;}
    .stApp .result {padding: 1rem; border-radius: 12px; font-size: 1.3rem; font-weight: 700;}
    .stApp .result-pass {border: 2px solid #000; background: #fff;}
    .stApp .result-pass, .stApp .result-pass * {color: #000 !important;}
    .stApp .result-fail {background: #000;}
    .stApp .result-fail, .stApp .result-fail * {color: #fff !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def train():
    df = pd.read_csv("data.csv")
    n_raw = len(df)
    df = df.drop_duplicates()  # augmented data has exact duplicates
    X = df[list(FEATURES)]
    y = (df["rating"] == "High").astype(int)  # High -> Pass, Low -> Fail
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(Xtr, ytr)
    pred = model.predict(Xte)
    report = classification_report(yte, pred, target_names=["Fail", "Pass"], output_dict=True)
    model.fit(X, y)  # refit on all data for the deployed model
    return model, df, n_raw, accuracy_score(yte, pred), report


model, df, n_raw, acc, report = train()

st.title("Student Success Predictor")
tab_predict, tab_about = st.tabs(["Predict", "About"])

with tab_predict:
    st.write("Enter a student's learning-activity record to predict whether they are likely to **pass or fail**.")
    with st.form("student"):
        c1, c2 = st.columns(2)
        vals = {}
        for i, (col, label) in enumerate(FEATURES.items()):
            with (c1 if i % 2 == 0 else c2):
                vals[col] = st.number_input(
                    label, min_value=0, value=int(df[col].median()), step=1,
                    help=f"Typical (median): {int(df[col].median())}; max in training data: {int(df[col].max())}",
                )
        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        x = pd.DataFrame([vals])[list(FEATURES)]
        p_pass = float(model.predict_proba(x)[0][1])
        if p_pass >= 0.5:
            st.markdown(
                f"<div class='result result-pass'>PASS: likely to succeed ({p_pass:.0%} confidence)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result result-fail'>FAIL: at risk ({1 - p_pass:.0%} confidence)</div>",
                unsafe_allow_html=True,
            )

with tab_about:
    st.header("About this app")
    st.write(
        "This tool helps the university identify students who may be at risk of failing, "
        "based on how they interact with the online learning platform. "
        "Enter a student's activity counts and the model returns a Pass or Fail prediction "
        "with a confidence score."
    )

    st.subheader("How it works")
    st.write(
        "A **Random Forest** classifier (300 decision trees) learns patterns from past student "
        "activity records. For a new student, each tree votes, and the share of votes for "
        "Pass becomes the confidence shown."
    )

    st.subheader("Data")
    st.write(
        f"The training file has {n_raw:,} records, of which {len(df):,} are unique "
        "(the dataset was augmented, so exact duplicates were removed before training "
        "to keep the accuracy estimate honest). The target is the `rating` column: "
        "**High = Pass**, **Low = Fail**."
    )

    st.subheader("Predictors")
    st.write(", ".join(FEATURES.values()) + ".")

    st.subheader("Model performance")
    st.write(f"Accuracy on a held-out 20% test set: **{acc:.1%}**.")
    perf = pd.DataFrame(report).T.loc[["Fail", "Pass"], ["precision", "recall", "f1-score", "support"]]
    rows = "".join(
        f"<tr><th>{name}</th><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
        f"<td>{r['f1-score']:.3f}</td><td>{int(r['support'])}</td></tr>"
        for name, r in perf.iterrows()
    )
    st.markdown(
        "<table class='perf'><thead><tr><th>Class</th><th>Precision</th><th>Recall</th>"
        f"<th>F1-score</th><th>Support</th></tr></thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )

    st.subheader("Limitations")
    st.write(
        "- Uses online activity only; grades, attendance, and personal circumstances are not included.\n"
        "- A prediction is a statistical estimate, not a verdict about any individual.\n"
        "- It should support, not replace, advising and instructor judgment."
    )
    st.caption("Built with Python, scikit-learn, and Streamlit.")
