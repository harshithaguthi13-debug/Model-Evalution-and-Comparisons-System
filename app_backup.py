import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Evaluation System",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Model Evaluation and Comparison System")

st.write(
    "A machine learning system for evaluating and comparing "
    "classification models using standard performance metrics."
)

st.markdown("---")


# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("📁 Dataset Selection")

st.write(
    "Upload a CSV dataset and select the target column "
    "that you want the machine learning models to predict."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "📌 Please upload a CSV dataset to start model evaluation."
    )

    st.markdown(
        """
        ### Recommended CSV format

        Your CSV should contain:

        - Multiple feature columns
        - One target column
        - At least 2 target classes

        **Example:**

        | sepal_length | sepal_width | petal_length | petal_width | target |
        |---:|---:|---:|---:|---|
        | 5.1 | 3.5 | 1.4 | 0.2 | Setosa |
        | 4.9 | 3.0 | 1.4 | 0.2 | Setosa |
        | 6.3 | 3.3 | 6.0 | 2.5 | Virginica |
        """
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    uploaded_df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"❌ Unable to read the CSV file: {e}"
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if uploaded_df.empty:

    st.error("❌ The uploaded CSV file is empty.")

    st.stop()


# ============================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

empty_columns = uploaded_df.columns[
    uploaded_df.isnull().all()
].tolist()

if empty_columns:

    uploaded_df = uploaded_df.drop(
        columns=empty_columns
    )


# ============================================================
# UPLOAD SUCCESS
# ============================================================

st.success(
    "✅ CSV dataset uploaded successfully!"
)


# ============================================================
# DATASET PREVIEW
# ============================================================

st.header("📊 Dataset Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        uploaded_df.shape[0]
    )

with col2:

    st.metric(
        "Columns",
        uploaded_df.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        int(uploaded_df.isnull().sum().sum())
    )

with col4:

    st.metric(
        "Duplicate Rows",
        int(uploaded_df.duplicated().sum())
    )


st.subheader("📋 Dataset Preview")

st.dataframe(
    uploaded_df.head(30),
    use_container_width=True
)


# ============================================================
# TARGET COLUMN SELECTION
# ============================================================

st.header("🎯 Target Column")

columns = uploaded_df.columns.tolist()


# Automatically select target if available
if "target" in columns:

    default_index = columns.index("target")

else:

    default_index = len(columns) - 1


target_column = st.selectbox(
    "Select the column that the model should predict:",
    columns,
    index=default_index
)


st.success(
    f"🎯 Selected target column: **{target_column}**"
)


# ============================================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================================

before_rows = len(uploaded_df)

uploaded_df = uploaded_df.dropna(
    subset=[target_column]
).copy()

after_rows = len(uploaded_df)

if before_rows != after_rows:

    st.warning(
        f"⚠️ Removed {before_rows - after_rows} rows "
        f"with missing target values."
    )


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = uploaded_df.drop(
    columns=[target_column]
).copy()

y = uploaded_df[target_column].copy()


# ============================================================
# TARGET VALIDATION
# ============================================================

number_of_classes = y.nunique()

st.subheader("🔍 Target Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Number of Features",
        X.shape[1]
    )

with col2:

    st.metric(
        "Number of Classes",
        number_of_classes
    )


st.write("**Target classes:**")

st.write(
    list(y.unique())
)


# ------------------------------------------------------------
# CHECK TARGET CLASSES
# ------------------------------------------------------------

if number_of_classes < 2:

    st.error(
        "❌ The selected target must contain at least "
        "two different classes."
    )

    st.stop()


# ------------------------------------------------------------
# CHECK FOR CONTINUOUS TARGET
# ------------------------------------------------------------

if pd.api.types.is_numeric_dtype(y):

    unique_ratio = y.nunique() / len(y)

    if unique_ratio > 0.5:

        st.error(
            "❌ The selected target appears to contain "
            "continuous numerical values."
        )

        st.info(
            "This project performs classification. "
            "Please select a categorical/discrete target column "
            "such as class, category, species, label, etc."
        )

        st.stop()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

st.subheader("📊 Target Class Distribution")

target_distribution = (
    y.value_counts()
    .rename_axis("Class")
    .reset_index(name="Count")
)

st.dataframe(
    target_distribution,
    use_container_width=True
)


# ============================================================
# CHECK DATASET SIZE
# ============================================================

if len(uploaded_df) < 20:

    st.warning(
        "⚠️ This dataset contains fewer than 20 rows. "
        "Model evaluation may not be very reliable."
    )


# ============================================================
# FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


st.subheader("🧩 Feature Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Numeric Features",
        len(numeric_features)
    )

with col2:

    st.metric(
        "Categorical Features",
        len(categorical_features)
    )


# ============================================================
# REMOVE UNSUITABLE FEATURES
# ============================================================

# Remove columns containing only one unique value
constant_columns = [
    column
    for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

if constant_columns:

    X = X.drop(
        columns=constant_columns
    )

    numeric_features = [
        column
        for column in numeric_features
        if column not in constant_columns
    ]

    categorical_features = [
        column
        for column in categorical_features
        if column not in constant_columns
    ]

    st.warning(
        "⚠️ Constant columns were removed: "
        + ", ".join(constant_columns)
    )


# ============================================================
# CHECK FEATURES
# ============================================================

if X.shape[1] == 0:

    st.error(
        "❌ No usable feature columns were found."
    )

    st.stop()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


transformers = []


if numeric_features:

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    )


if categorical_features:

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

st.header("⚙️ Model Configuration")


# ------------------------------------------------------------
# Check minimum class size
# ------------------------------------------------------------

class_counts = y.value_counts()

minimum_class_count = class_counts.min()


# ------------------------------------------------------------
# k VALUE
# ------------------------------------------------------------

max_k = min(
    15,
    max(
        1,
        int(len(X) * 0.8)
    )
)


default_k = min(
    5,
    max_k
)


k = st.slider(
    "Select k value for kNN",
    min_value=1,
    max_value=max_k,
    value=default_k
)


st.write(
    f"Selected k value: **{k}**"
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

except ValueError:

    st.warning(
        "⚠️ Stratified split could not be performed. "
        "Using a normal train/test split."
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


# ============================================================
# CREATE MODELS
# ============================================================

knn_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        KNeighborsClassifier(
            n_neighbors=k
        )
    )
])


svm_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        SVC(
            kernel="rbf"
        )
    )
])


# ============================================================
# MODEL TRAINING
# ============================================================

st.header("🚀 Model Training")


# ------------------------------------------------------------
# kNN
# ------------------------------------------------------------

with st.spinner("Training kNN model..."):

    knn_model.fit(
        X_train,
        y_train
    )

    knn_predictions = knn_model.predict(
        X_test
    )


st.success(
    "✅ kNN model trained successfully!"
)


# ------------------------------------------------------------
# SVM
# ------------------------------------------------------------

with st.spinner("Training SVM model..."):

    svm_model.fit(
        X_train,
        y_train
    )

    svm_predictions = svm_model.predict(
        X_test
    )


st.success(
    "✅ SVM model trained successfully!"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predictions
):

    return {

        "Accuracy":
            accuracy_score(
                actual,
                predictions
            ),

        "Precision":
            precision_score(
                actual,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "Recall":
            recall_score(
                actual,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "F1-Score":
            f1_score(
                actual,
                predictions,
                average="weighted",
                zero_division=0
            )
    }


# ============================================================
# CALCULATE MODEL METRICS
# ============================================================

knn_metrics = calculate_metrics(
    y_test,
    knn_predictions
)


svm_metrics = calculate_metrics(
    y_test,
    svm_predictions
)


# ============================================================
# CROSS VALIDATION
# ============================================================

st.header("🔄 5-Fold Cross-Validation")


if minimum_class_count >= 5:

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    with st.spinner(
        "Running 5-fold cross-validation..."
    ):

        knn_cv_scores = cross_val_score(
            knn_model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )


        svm_cv_scores = cross_val_score(
            svm_model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )


    knn_cv_mean = knn_cv_scores.mean()

    svm_cv_mean = svm_cv_scores.mean()


    # ========================================================
    # CV SUMMARY
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🤖 kNN CV Accuracy",
            f"{knn_cv_mean * 100:.2f}%"
        )

    with col2:

        st.metric(
            "🧠 SVM CV Accuracy",
            f"{svm_cv_mean * 100:.2f}%"
        )


    # ========================================================
    # FOLD RESULTS
    # ========================================================

    cv_results = pd.DataFrame({

        "Fold": [
            1,
            2,
            3,
            4,
            5
        ],

        "kNN Accuracy": (
            knn_cv_scores * 100
        ).round(2),

        "SVM Accuracy": (
            svm_cv_scores * 100
        ).round(2)
    })


    st.subheader(
        "📋 Individual Fold Results"
    )


    st.dataframe(
        cv_results,
        use_container_width=True
    )


    # ========================================================
    # CV CHART
    # ========================================================

    st.subheader(
        "📊 Cross-Validation Accuracy Comparison"
    )


    fig, ax = plt.subplots()


    ax.plot(
        cv_results["Fold"],
        cv_results["kNN Accuracy"],
        marker="o",
        label="kNN"
    )


    ax.plot(
        cv_results["Fold"],
        cv_results["SVM Accuracy"],
        marker="o",
        label="SVM"
    )


    ax.set_xlabel(
        "Fold"
    )


    ax.set_ylabel(
        "Accuracy (%)"
    )


    ax.set_title(
        "5-Fold Cross-Validation Accuracy"
    )


    ax.legend()

    ax.grid(True)


    st.pyplot(fig)

    plt.close(fig)


else:

    st.warning(
        "⚠️ 5-fold cross-validation is not available "
        "because at least one target class contains fewer "
        "than 5 samples."
    )

    knn_cv_mean = None
    svm_cv_mean = None


# ============================================================
# MODEL EVALUATION
# ============================================================

st.header("📈 Model Evaluation")


# ============================================================
# kNN RESULTS
# ============================================================

st.subheader("🤖 kNN Results")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{knn_metrics['Accuracy'] * 100:.2f}%"
    )


with col2:

    st.metric(
        "Precision",
        f"{knn_metrics['Precision'] * 100:.2f}%"
    )


with col3:

    st.metric(
        "Recall",
        f"{knn_metrics['Recall'] * 100:.2f}%"
    )


with col4:

    st.metric(
        "F1-Score",
        f"{knn_metrics['F1-Score'] * 100:.2f}%"
    )


# ============================================================
# SVM RESULTS
# ============================================================

st.subheader("🧠 SVM Results")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{svm_metrics['Accuracy'] * 100:.2f}%"
    )


with col2:

    st.metric(
        "Precision",
        f"{svm_metrics['Precision'] * 100:.2f}%"
    )


with col3:

    st.metric(
        "Recall",
        f"{svm_metrics['Recall'] * 100:.2f}%"
    )


with col4:

    st.metric(
        "F1-Score",
        f"{svm_metrics['F1-Score'] * 100:.2f}%"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("🏆 Model Comparison")


comparison = pd.DataFrame({

    "Model": [
        "kNN",
        "SVM"
    ],

    "Accuracy": [
        knn_metrics["Accuracy"],
        svm_metrics["Accuracy"]
    ],

    "Precision": [
        knn_metrics["Precision"],
        svm_metrics["Precision"]
    ],

    "Recall": [
        knn_metrics["Recall"],
        svm_metrics["Recall"]
    ],

    "F1-Score": [
        knn_metrics["F1-Score"],
        svm_metrics["F1-Score"]
    ],

    "CV Accuracy": [
        knn_cv_mean if knn_cv_mean is not None else float("nan"),
        svm_cv_mean if svm_cv_mean is not None else float("nan")
    ]
})


# ============================================================
# DISPLAY PERCENTAGES
# ============================================================

comparison_display = comparison.copy()


metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "CV Accuracy"
]


for column in metric_columns:

    comparison_display[column] = (
        comparison_display[column] * 100
    ).round(2)


comparison_display = comparison_display.set_index(
    "Model"
)


st.dataframe(
    comparison_display,
    use_container_width=True
)


# ============================================================
# PERFORMANCE CHART
# ============================================================

st.subheader(
    "📊 Model Performance Comparison"
)


chart_data = comparison_display[
    metric_columns
]


st.bar_chart(
    chart_data
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

st.header("🔍 Confusion Matrix")


class_labels = sorted(
    y.unique(),
    key=lambda value: str(value)
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# kNN CONFUSION MATRIX
# ------------------------------------------------------------

with col1:

    st.subheader(
        "🤖 kNN Confusion Matrix"
    )


    knn_cm = confusion_matrix(
        y_test,
        knn_predictions,
        labels=class_labels
    )


    knn_cm_df = pd.DataFrame(

        knn_cm,

        index=[
            f"Actual {label}"
            for label in class_labels
        ],

        columns=[
            f"Predicted {label}"
            for label in class_labels
        ]
    )


    st.dataframe(
        knn_cm_df,
        use_container_width=True
    )


# ------------------------------------------------------------
# SVM CONFUSION MATRIX
# ------------------------------------------------------------

with col2:

    st.subheader(
        "🧠 SVM Confusion Matrix"
    )


    svm_cm = confusion_matrix(
        y_test,
        svm_predictions,
        labels=class_labels
    )


    svm_cm_df = pd.DataFrame(

        svm_cm,

        index=[
            f"Actual {label}"
            for label in class_labels
        ],

        columns=[
            f"Predicted {label}"
            for label in class_labels
        ]
    )


    st.dataframe(
        svm_cm_df,
        use_container_width=True
    )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

st.header("🔎 Model Predictions")


prediction_results = pd.DataFrame({

    "Actual": y_test.values,

    "kNN Prediction": knn_predictions,

    "SVM Prediction": svm_predictions
})


st.dataframe(
    prediction_results,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

st.header("📑 Classification Reports")


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "🤖 kNN Classification Report"
    )


    knn_report = classification_report(

        y_test,

        knn_predictions,

        output_dict=True,

        zero_division=0
    )


    knn_report_df = pd.DataFrame(
        knn_report
    ).transpose()


    st.dataframe(
        knn_report_df.round(3),
        use_container_width=True
    )


with col2:

    st.subheader(
        "🧠 SVM Classification Report"
    )


    svm_report = classification_report(

        y_test,

        svm_predictions,

        output_dict=True,

        zero_division=0
    )


    svm_report_df = pd.DataFrame(
        svm_report
    ).transpose()


    st.dataframe(
        svm_report_df.round(3),
        use_container_width=True
    )


# ============================================================
# BEST MODEL
# ============================================================

best_model = comparison.loc[
    comparison["F1-Score"].idxmax(),
    "Model"
]


best_f1 = comparison.loc[
    comparison["F1-Score"].idxmax(),
    "F1-Score"
]


# ============================================================
# BEST CV MODEL
# ============================================================

if knn_cv_mean is not None:

    best_cv_model = comparison.loc[
        comparison["CV Accuracy"].idxmax(),
        "Model"
    ]


    best_cv_score = comparison.loc[
        comparison["CV Accuracy"].idxmax(),
        "CV Accuracy"
    ]

else:

    best_cv_model = None

    best_cv_score = None


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

st.header("🏆 Final Recommendation")


st.success(
    f"🏆 Best performing model based on F1-Score: "
    f"**{best_model}** "
    f"({best_f1 * 100:.2f}%)"
)


if best_cv_model is not None:

    st.info(
        f"🔄 Best model based on 5-Fold "
        f"Cross-Validation: "
        f"**{best_cv_model}** "
        f"({best_cv_score * 100:.2f}%)"
    )


# ============================================================
# CONCLUSION
# ============================================================

st.header("📝 Conclusion")


if best_cv_model is not None:

    if best_model == best_cv_model:

        st.success(

            f"Based on the evaluation results, "
            f"**{best_model}** is the recommended model. "
            f"It achieved the best performance according "
            f"to both F1-Score and Cross-Validation accuracy."
        )

    else:

        st.warning(

            f"The evaluation methods produced different "
            f"best models. **{best_model}** performed best "
            f"according to F1-Score, while "
            f"**{best_cv_model}** performed best during "
            f"cross-validation."
        )

else:

    st.info(

        f"Based on the F1-Score, "
        f"**{best_model}** is the recommended model."
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.header("⬇️ Download Results")


download_data = comparison_display.reset_index()


csv_data = download_data.to_csv(
    index=False
)


st.download_button(

    label="📥 Download Model Comparison CSV",

    data=csv_data,

    file_name="model_comparison_results.csv",

    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(
    "Model Evaluation and Comparison System | "
    "Machine Learning Project"
)