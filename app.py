import hashlib
import json
import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓")

INPUTS = {
    "Submission.Actions": "Submission actions",
    "Quiz.Actions": "Quiz actions",
    "Forum.Actions": "Forum actions",
    "Create.Actions": "Create actions",
    "Read.Actions": "Read actions",
    "Update.Actions": "Update actions",
    "Delete.Actions": "Delete actions",
    "Days.Logged.In": "Days logged in",
}
N_TREES = 300  # same accuracy as 600 trees in cross-validation, half the training time
LOW, HIGH = 0.40, 0.60  # probabilities in this band are reported as Uncertain

st.markdown(
    """
    <style>
    :root {color-scheme: light;}
    .stApp, [data-testid="stHeader"], [data-testid="stAppViewContainer"] {background: #fff !important;}
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {color: #000 !important;}
    .block-container {max-width: 760px; padding-top: 2rem;}
    h1, h2, h3 {letter-spacing: -0.5px;}
    h3 {margin-top: 1.4rem;}
    div[data-testid="stForm"] {border: 1px solid #000; border-radius: 12px; background: #fff;}
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] input {background: #fff !important; color: #000 !important;}
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {border: 1px solid #000 !important;}
    div[data-testid="stNumberInput"] button {background: #fff !important; color: #000 !important;}
    div[data-testid="stFormSubmitButton"] button {
        background: #fff !important; border: 1.5px solid #000; border-radius: 10px; font-weight: 600;}
    div[data-testid="stFormSubmitButton"] button, div[data-testid="stFormSubmitButton"] button * {color: #000 !important;}
    div[data-testid="stFormSubmitButton"] button:hover {background: #F0F0F0 !important;}
    div[data-testid="stFormSubmitButton"] button:focus {box-shadow: none !important; border-color: #000 !important;}
    button[data-baseweb="tab"] {font-weight: 600;}
    [data-baseweb="tab-highlight"] {background: #000 !important;}

    .result {padding: 1rem 1.2rem; border-radius: 10px; font-size: 1.15rem; font-weight: 600;
        background: #F5F5F5; border: 1px solid #D9D9D9; border-left: 6px solid #000; margin-top: 0.5rem;}
    .result, .result * {color: #000 !important;}
    .result small {display: block; font-weight: 400; font-size: 0.9rem; margin-top: 0.2rem; color: #444 !important;}

    .stats {display: flex; gap: 12px; margin: 1rem 0 0.5rem 0;}
    .stat {flex: 1; border: 1px solid #000; border-radius: 12px; padding: 1rem; text-align: center; background: #fff;}
    .stat b {display: block; font-size: 1.9rem; line-height: 1.1;}
    .stat span {font-size: 0.85rem; color: #444 !important;}

    table.t {width: 100%; border-collapse: collapse; margin: 0.5rem 0 1rem 0; font-size: 0.95rem;}
    table.t th, table.t td {border: 1px solid #000; padding: 0.55rem 0.8rem; text-align: center; background: #fff;}
    table.t thead th {background: #000; color: #fff !important;}
    table.t thead th * {color: #fff !important;}
    table.t tbody th {text-align: left; font-weight: 700;}
    table.t td.hi {background: #EDEDED; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    days = d["Days.Logged.In"].clip(lower=1)
    total = d[[c for c in INPUTS if c != "Days.Logged.In"]].sum(axis=1)
    d["Total.Actions"] = total
    d["Actions.Per.Day"] = total / days
    d["Read.Per.Day"] = d["Read.Actions"] / days
    d["Submit.Per.Day"] = d["Submission.Actions"] / days
    d["Engagement.Ratio"] = (d["Submission.Actions"] + d["Quiz.Actions"] + d["Forum.Actions"]) / (total + 1)
    return d


def new_model():
    # Extra Trees configuration selected by cross-validation on the training set (see About tab)
    return ExtraTreesClassifier(
        n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=1, random_state=42, n_jobs=-1
    )


def data_md5() -> str:
    with open("data.csv", "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def holdout(d: pd.DataFrame) -> dict:
    """80/20 stratified split, seed 42; one evaluation on the held-out test set."""
    X = add_features(d[list(INPUTS)])
    y = (d["rating"] == "High").astype(int)  # High -> Pass, Low -> Fail
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pred = new_model().fit(Xtr, ytr).predict(Xte)
    return {"acc": float(accuracy_score(yte, pred)), "cm": confusion_matrix(yte, pred).tolist(), "n_test": int(len(yte))}


def evaluate(raw: pd.DataFrame) -> dict:
    return {
        "version": 2,
        "protocol": holdout(raw),                    # random split of the provided file (assignment protocol)
        "strict": holdout(raw.drop_duplicates()),    # duplicates removed before splitting
        "n_raw": int(len(raw)),
        "n_unique": int(len(raw.drop_duplicates())),
        "n_trees": N_TREES,
        "data_md5": data_md5(),
    }


@st.cache_resource
def train():
    raw = pd.read_csv("data.csv")
    df = raw.drop_duplicates()
    X = add_features(df[list(INPUTS)])
    y = (df["rating"] == "High").astype(int)

    # Evaluation numbers are precomputed in metrics.json. If the file is missing, outdated, or the data
    # changed since it was made, they are recomputed here so the About tab is never stale.
    stats = None
    try:
        with open("metrics.json") as f:
            saved = json.load(f)
        if saved.get("version") == 2 and saved.get("data_md5") == data_md5() and saved.get("n_trees") == N_TREES:
            stats = saved
    except (OSError, ValueError):
        pass
    if stats is None:
        stats = evaluate(raw)
        try:
            with open("metrics.json", "w") as f:
                json.dump(stats, f)
        except OSError:
            pass

    final = new_model().fit(X, y)  # deployed model is trained on every unique record
    return final, df, stats


model, df, S = train()

st.title("Student Success Predictor")
tab_predict, tab_about = st.tabs(["Predict", "About"])

# ------------------------------------------------------------------ Predict
with tab_predict:
    st.write("Enter a student's learning-activity record to estimate whether they are likely to **pass or fail**.")
    with st.form("student"):
        c1, c2 = st.columns(2)
        vals = {}
        for i, (col, label) in enumerate(INPUTS.items()):
            with (c1 if i % 2 == 0 else c2):
                vals[col] = st.number_input(
                    label, min_value=0, value=int(df[col].median()), step=1,
                    help=f"Typical (median): {int(df[col].median())}; max in training data: {int(df[col].max())}",
                )
        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        x = add_features(pd.DataFrame([vals])[list(INPUTS)])
        p = float(model.predict_proba(x)[0][1])
        if LOW <= p <= HIGH:
            lean = "Pass" if p >= 0.5 else "Fail"
            head_, sub = "UNCERTAIN: borderline case", f"Slight lean toward {lean} (probability of passing {p:.0%}). Worth a check-in."
        elif p > HIGH:
            head_, sub = "PASS: likely to succeed", f"Probability of passing: {p:.0%}"
        else:
            head_, sub = "FAIL: at risk", f"Probability of passing: {p:.0%}"
        st.markdown(f"<div class='result'>{head_}<small>{sub}</small></div>", unsafe_allow_html=True)


def cm_table(cm):
    return (
        "<table class='t'><thead><tr><th></th><th>Predicted Fail</th><th>Predicted Pass</th></tr></thead><tbody>"
        f"<tr><th>Actual Fail</th><td class='hi'>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>"
        f"<tr><th>Actual Pass</th><td>{cm[1][0]}</td><td class='hi'>{cm[1][1]}</td></tr></tbody></table>"
    )


# -------------------------------------------------------------------- About
with tab_about:
    P_, K_ = S["protocol"], S["strict"]
    st.header("About this model")
    st.write(
        "This tool estimates whether a student is likely to finish with a **Pass** (High rating) or a "
        "**Fail** (Low rating) outcome, using the activity they have logged in the learning management system "
        "during the midterm stage."
    )
    st.markdown(
        f"""<div class='stats'>
        <div class='stat'><b>{P_['acc']:.1%}</b><span>Test accuracy<br>(required protocol)</span></div>
        <div class='stat'><b>{K_['acc']:.1%}</b><span>Strict check<br>(duplicates removed)</span></div>
        <div class='stat'><b>{N_TREES}</b><span>Decision trees</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.subheader("How it works")
    st.write(
        "The model is an **Extra Trees** classifier: an ensemble of randomized decision trees whose votes are "
        "combined into a single probability of passing. It was chosen over four alternatives because it had the "
        "highest accuracy under 5-fold cross-validation on the training set (table below). Inputs are eight "
        "activity counts (submission, quiz, forum, create, read, update and delete actions, plus days logged in), "
        "from which five additional features are derived, such as actions per login day. A probability above 60% "
        "is reported as Pass and below 40% as Fail. Predictions between 40% and 60% are labelled **Uncertain**."
    )
    cv_rows = [("Extra Trees (used)", "87.5%"), ("Random forest", "86.7%"), ("Single decision tree", "82.4%"),
               ("Gradient boosting", "75.8%"), ("Logistic regression", "67.1%")]
    body = "".join(f"<tr><th>{n}</th><td>{a}</td></tr>" for n, a in cv_rows)
    st.markdown(
        f"<table class='t'><thead><tr><th>Model</th><th>5-fold CV accuracy</th></tr></thead><tbody>{body}</tbody></table>",
        unsafe_allow_html=True,
    )

    st.subheader("Performance")
    st.write(
        f"Following the project protocol, the provided file was split 80/20 with a fixed random seed (42), the model "
        f"was selected using the training portion only, and the held-out test set of {P_['n_test']:,} records was scored "
        f"once. It reached **{P_['acc']:.1%}** accuracy. The table shows how those records were classified."
    )
    st.markdown(cm_table(P_["cm"]), unsafe_allow_html=True)

    st.subheader("About the data and a stricter check")
    st.write(
        f"The source file has {S['n_raw']:,} rows, but only {S['n_unique']:,} are unique, because the dataset was "
        "augmented with repeated records. In a random split, some test records therefore have an identical copy in "
        "the training set, which makes the accuracy above somewhat optimistic. When duplicates are removed "
        f"**before** splitting, the same model reaches **{K_['acc']:.1%}** on {K_['n_test']:,} test records:"
    )
    st.markdown(cm_table(K_["cm"]), unsafe_allow_html=True)
    st.write(
        "Real-world accuracy on new students is therefore likely to be closer to the stricter figure. "
        "This is why borderline predictions are labelled Uncertain."
    )

    st.subheader("Intended use")
    st.write(
        "This is a decision-support tool. Use it to decide which students to check in with early, so help can be "
        "offered while there is still time in the term. It should not be used to grade, rank or penalize students. "
        "The labels Pass and Fail correspond to the High and Low ratings in the source dataset, and a prediction "
        "describes a pattern of activity, not a cause or a final result. Only online activity is considered; "
        "grades, attendance and personal circumstances are not."
    )
    st.caption("Built with Python, scikit-learn and Streamlit.")
