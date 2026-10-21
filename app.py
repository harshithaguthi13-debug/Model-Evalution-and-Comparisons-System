import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
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
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Evaluation and Comparison System",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Model Evaluation and Comparison System")

st.write(
    "A machine learning system for evaluating, comparing and "
    "visualizing classification and regression models."
)

st.markdown("---")


# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("📁 Dataset Selection")

st.write(
    "Upload a CSV dataset and select the machine learning "
    "task you want to perform."
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
        "📌 Please upload a CSV dataset to start "
        "the model evaluation."
    )

    st.markdown(
        """
        ### Recommended Dataset Format

        Your CSV should contain:

        - Multiple feature columns
        - One target column
        - For Classification: categorical/discrete target
        - For Regression: numerical continuous target

        ### Classification Example

        | sepal_length | sepal_width | petal_length | petal_width | species |
        |---:|---:|---:|---:|---|
        | 5.1 | 3.5 | 1.4 | 0.2 | Setosa |
        | 4.9 | 3.0 | 1.4 | 0.2 | Setosa |
        | 6.3 | 3.3 | 6.0 | 2.5 | Virginica |

        ### Regression Example

        | area | bedrooms | age | price |
        |---:|---:|---:|---:|
        | 1200 | 2 | 10 | 250000 |
        | 1800 | 3 | 5 | 350000 |
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

    st.error(
        "❌ The uploaded CSV file is empty."
    )

    st.stop()


if uploaded_df.shape[1] < 2:

    st.error(
        "❌ Dataset must contain at least one feature "
        "column and one target column."
    )

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

    st.warning(
        "⚠️ Completely empty columns were removed: "
        + ", ".join(empty_columns)
    )


# ============================================================
# SUCCESS
# ============================================================

st.success(
    "✅ CSV dataset uploaded successfully!"
)


# ============================================================
# DATASET INFORMATION
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
        int(
            uploaded_df.isnull()
            .sum()
            .sum()
        )
    )


with col4:

    st.metric(
        "Duplicate Rows",
        int(
            uploaded_df.duplicated()
            .sum()
        )
    )


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("📋 Dataset Preview")


st.dataframe(
    uploaded_df.head(30),
    use_container_width=True
)


# ============================================================
# MACHINE LEARNING TASK
# ============================================================

st.header("🎯 Machine Learning Task")


task = st.radio(

    "Select the type of machine learning problem:",

    [
        "Classification",
        "Regression"
    ],

    horizontal=True
)


if task == "Classification":

    st.info(
        "📘 Classification predicts discrete categories "
        "such as species, class, type or category."
    )

else:

    st.info(
        "📈 Regression predicts continuous numerical values "
        "such as price, salary, marks, sales or temperature."
    )


# ============================================================
# TARGET COLUMN
# ============================================================

st.header("🎯 Target Column")


columns = uploaded_df.columns.tolist()


# Automatically choose target
# If "target" exists, select it.
# Otherwise select the last column.

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
# REMOVE MISSING TARGET VALUES
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
# CHECK MINIMUM DATA
# ============================================================

if len(uploaded_df) < 5:

    st.error(
        "❌ At least 5 valid rows are required."
    )

    st.stop()


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = uploaded_df.drop(
    columns=[target_column]
).copy()


y = uploaded_df[target_column].copy()


# ============================================================
# REMOVE EMPTY FEATURES
# ============================================================

empty_feature_columns = X.columns[
    X.isnull().all()
].tolist()


if empty_feature_columns:

    X = X.drop(
        columns=empty_feature_columns
    )

    st.warning(
        "⚠️ Completely empty feature columns were removed: "
        + ", ".join(empty_feature_columns)
    )


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [

    column

    for column in X.columns

    if X[column].nunique(
        dropna=True
    ) <= 1

]


if constant_columns:

    X = X.drop(
        columns=constant_columns
    )

    st.warning(
        "⚠️ Constant feature columns were removed: "
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
# DATASET DETAILS
# ============================================================

st.header("🔍 Dataset Details")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Usable Features",
        X.shape[1]
    )


with col2:

    st.metric(
        "Samples",
        X.shape[0]
    )


with col3:

    if task == "Classification":

        st.metric(
            "Number of Classes",
            y.nunique()
        )

    else:

        st.metric(
            "Target Type",
            str(y.dtype)
        )


# ============================================================
# ============================================================
# CLASSIFICATION
# ============================================================
# ============================================================

if task == "Classification":

    st.header("🔵 Classification Analysis")


    # ========================================================
    # CLASSIFICATION TARGET VALIDATION
    # ========================================================

    number_of_classes = y.nunique()


    if number_of_classes < 2:

        st.error(
            "❌ Classification requires at least two "
            "different target classes."
        )

        st.stop()


    # --------------------------------------------------------
    # DETECT CONTINUOUS NUMERICAL TARGET
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(y):

        unique_ratio = (
            y.nunique() / len(y)
        )

        if unique_ratio > 0.5:

            st.error(
                "❌ The selected target appears to contain "
                "continuous numerical values."
            )

            st.info(
                "👉 Change the task above to **Regression** "
                "or select a categorical target column."
            )

            st.stop()


    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    st.subheader("🎯 Classification Target Information")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Number of Classes",
            number_of_classes
        )


    with col2:

        st.metric(
            "Samples",
            len(y)
        )


    st.write(
        "**Target Classes:**"
    )


    st.write(
        list(y.unique())
    )


    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Target Class Distribution"
    )


    target_distribution = (

        y.value_counts()

        .rename_axis("Class")

        .reset_index(
            name="Count"
        )

    )


    st.dataframe(
        target_distribution,
        use_container_width=True
    )


    # ========================================================
    # FEATURE TYPES
    # ========================================================

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()


    categorical_features = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()


    st.subheader(
        "🧩 Feature Information"
    )


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


    # ========================================================
    # PREPROCESSING
    # ========================================================

    transformers = []


    if numeric_features:

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


        transformers.append(

            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )

        )


    if categorical_features:

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


        transformers.append(

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )

        )


    if not transformers:

        st.error(
            "❌ No usable feature columns were found."
        )

        st.stop()


    preprocessor = ColumnTransformer(
        transformers=transformers
    )


    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    st.header(
        "⚙️ Classification Model Configuration"
    )


    class_counts = y.value_counts()


    minimum_class_count = class_counts.min()


    max_k = min(
        15,
        max(
            1,
            int(
                len(X) * 0.8
            )
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


    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

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
            "Using normal train/test split."
        )


        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42
        )


    # ========================================================
    # CREATE kNN
    # ========================================================

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


    # ========================================================
    # CREATE SVM
    # ========================================================

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


    # ========================================================
    # MODEL TRAINING
    # ========================================================

    st.header(
        "🚀 Model Training"
    )


    # --------------------------------------------------------
    # kNN
    # --------------------------------------------------------

    with st.spinner(
        "Training kNN model..."
    ):

        knn_model.fit(
            X_train,
            y_train
        )


        knn_predictions = (
            knn_model.predict(
                X_test
            )
        )


    st.success(
        "✅ kNN model trained successfully!"
    )


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    with st.spinner(
        "Training SVM model..."
    ):

        svm_model.fit(
            X_train,
            y_train
        )


        svm_predictions = (
            svm_model.predict(
                X_test
            )
        )


    st.success(
        "✅ SVM model trained successfully!"
    )


    # ========================================================
    # METRIC FUNCTION
    # ========================================================

    def calculate_classification_metrics(
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


    # ========================================================
    # MODEL METRICS
    # ========================================================

    knn_metrics = calculate_classification_metrics(

        y_test,
        knn_predictions

    )


    svm_metrics = calculate_classification_metrics(

        y_test,
        svm_predictions

    )


    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    st.header(
        "🔄 5-Fold Cross-Validation"
    )


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


        knn_cv_mean = (
            knn_cv_scores.mean()
        )


        svm_cv_mean = (
            svm_cv_scores.mean()
        )


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


        # ====================================================
        # CV TABLE
        # ====================================================

        cv_results = pd.DataFrame({

            "Fold": [
                1,
                2,
                3,
                4,
                5
            ],

            "kNN Accuracy":

                (
                    knn_cv_scores * 100
                ).round(2),

            "SVM Accuracy":

                (
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


        # ====================================================
        # CV GRAPH
        # ====================================================

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


        ax.grid(
            True
        )


        st.pyplot(fig)


        plt.close(fig)


    else:

        st.warning(

            "⚠️ 5-fold cross-validation is not available "
            "because at least one class contains fewer "
            "than 5 samples."
        )


        knn_cv_mean = None

        svm_cv_mean = None


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    st.header(
        "📈 Model Evaluation"
    )


    # ========================================================
    # kNN RESULTS
    # ========================================================

    st.subheader(
        "🤖 kNN Results"
    )


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


    # ========================================================
    # SVM RESULTS
    # ========================================================

    st.subheader(
        "🧠 SVM Results"
    )


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


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.header(
        "🏆 Classification Model Comparison"
    )


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

            knn_cv_mean
            if knn_cv_mean is not None
            else np.nan,

            svm_cv_mean
            if svm_cv_mean is not None
            else np.nan

        ]

    })


    comparison_display = (
        comparison.copy()
    )


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


    comparison_display = (
        comparison_display
        .set_index("Model")
    )


    st.dataframe(

        comparison_display,

        use_container_width=True
    )


    # ========================================================
    # PERFORMANCE CHART
    # ========================================================

    st.subheader(
        "📊 Model Performance Comparison"
    )


    st.bar_chart(
        comparison_display[
            metric_columns
        ]
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.header(
        "🔍 Confusion Matrix"
    )


    class_labels = sorted(

        y.unique(),

        key=lambda value: str(value)

    )


    col1, col2 = st.columns(2)


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


    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.header(
        "🔎 Model Predictions"
    )


    prediction_results = pd.DataFrame({

        "Actual":

            y_test.values,

        "kNN Prediction":

            knn_predictions,

        "SVM Prediction":

            svm_predictions

    })


    st.dataframe(

        prediction_results,

        use_container_width=True

    )


    # ========================================================
    # CLASSIFICATION REPORTS
    # ========================================================

    st.header(
        "📑 Classification Reports"
    )


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


        st.dataframe(

            pd.DataFrame(
                knn_report
            ).transpose().round(3),

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


        st.dataframe(

            pd.DataFrame(
                svm_report
            ).transpose().round(3),

            use_container_width=True

        )


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model = comparison.loc[

        comparison["F1-Score"].idxmax(),

        "Model"

    ]


    best_f1 = comparison.loc[

        comparison["F1-Score"].idxmax(),

        "F1-Score"

    ]


    st.header(
        "🏆 Final Recommendation"
    )


    st.success(

        f"🏆 Best performing classification model "
        f"based on F1-Score: **{best_model}** "
        f"({best_f1 * 100:.2f}%)"

    )


    if knn_cv_mean is not None:

        best_cv_model = comparison.loc[

            comparison["CV Accuracy"].idxmax(),

            "Model"

        ]


        best_cv_score = comparison.loc[

            comparison["CV Accuracy"].idxmax(),

            "CV Accuracy"

        ]


        st.info(

            f"🔄 Best model based on 5-Fold "
            f"Cross-Validation: **{best_cv_model}** "
            f"({best_cv_score * 100:.2f}%)"

        )


    # ========================================================
    # CONCLUSION
    # ========================================================

    st.header(
        "📝 Classification Conclusion"
    )


    st.write(

        f"The classification models kNN and SVM were "
        f"trained and evaluated using Accuracy, Precision, "
        f"Recall and F1-Score. Based on the F1-Score, "
        f"**{best_model}** achieved the best classification "
        f"performance."

    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.header(
        "⬇️ Download Classification Results"
    )


    download_data = (
        comparison_display
        .reset_index()
    )


    csv_data = (
        download_data
        .to_csv(index=False)
    )


    st.download_button(

        label="📥 Download Classification Results",

        data=csv_data,

        file_name="classification_model_results.csv",

        mime="text/csv"

    )


# ============================================================
# ============================================================
# REGRESSION
# ============================================================
# ============================================================

else:

    st.header(
        "📈 Regression Analysis"
    )


    st.write(

        "Regression predicts continuous numerical values "
        "using Linear Regression and evaluates the model "
        "using MAE, MSE, RMSE and R² Score."

    )


    # ========================================================
    # CONVERT TARGET TO NUMERIC
    # ========================================================

    numeric_target = pd.to_numeric(

        y,

        errors="coerce"

    )


    invalid_values = (
        numeric_target.isna().sum()
    )


    if invalid_values > 0:

        st.error(

            "❌ Regression requires a numeric target column."

        )


        st.info(

            "Please select a target containing continuous "
            "numerical values such as price, salary, marks, "
            "sales, temperature, etc."

        )


        st.stop()


    # Replace y with numeric target

    y = numeric_target


    # ========================================================
    # REGRESSION TARGET INFORMATION
    # ========================================================

    st.subheader(
        "🎯 Regression Target Information"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "Minimum",

            f"{y.min():.2f}"

        )


    with col2:

        st.metric(

            "Maximum",

            f"{y.max():.2f}"

        )


    with col3:

        st.metric(

            "Mean",

            f"{y.mean():.2f}"

        )


    with col4:

        st.metric(

            "Unique Values",

            y.nunique()

        )


    # ========================================================
    # FEATURE TYPES
    # ========================================================

    numeric_features = X.select_dtypes(

        include=["number"]

    ).columns.tolist()


    categorical_features = X.select_dtypes(

        include=[
            "object",
            "category",
            "bool"
        ]

    ).columns.tolist()


    st.subheader(
        "🧩 Regression Feature Information"
    )


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


    if (
        len(numeric_features) == 0
        and
        len(categorical_features) == 0
    ):

        st.error(

            "❌ No usable feature columns were found."

        )

        st.stop()


    # ========================================================
    # PREPROCESSING
    # ========================================================

    transformers = []


    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    if numeric_features:

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


        transformers.append(

            (
                "numeric",

                numeric_pipeline,

                numeric_features

            )

        )


    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    if categorical_features:

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


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )


    # ========================================================
    # LINEAR REGRESSION MODEL
    # ========================================================

    regression_model = Pipeline([

        (
            "preprocessor",

            preprocessor

        ),

        (
            "regressor",

            LinearRegression()

        )

    ])


    # ========================================================
    # MODEL TRAINING
    # ========================================================

    st.header(
        "🚀 Regression Model Training"
    )


    with st.spinner(

        "Training Linear Regression model..."

    ):

        regression_model.fit(

            X_train,

            y_train

        )


        regression_predictions = (

            regression_model.predict(

                X_test

            )

        )


    st.success(

        "✅ Linear Regression model trained successfully!"

    )


    # ========================================================
    # REGRESSION METRICS
    # ========================================================

    mae = mean_absolute_error(

        y_test,

        regression_predictions

    )


    mse = mean_squared_error(

        y_test,

        regression_predictions

    )


    rmse = np.sqrt(
        mse
    )


    r2 = r2_score(

        y_test,

        regression_predictions

    )


    # ========================================================
    # DISPLAY REGRESSION METRICS
    # ========================================================

    st.header(
        "📊 Regression Evaluation"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "MAE",

            f"{mae:.4f}"

        )


    with col2:

        st.metric(

            "MSE",

            f"{mse:.4f}"

        )


    with col3:

        st.metric(

            "RMSE",

            f"{rmse:.4f}"

        )


    with col4:

        st.metric(

            "R² Score",

            f"{r2:.4f}"

        )


    # ========================================================
    # REGRESSION CROSS VALIDATION
    # ========================================================

    st.header(
        "🔄 Regression 5-Fold Cross-Validation"
    )


    if len(X) >= 5:

        cv = KFold(

            n_splits=5,

            shuffle=True,

            random_state=42

        )


        with st.spinner(

            "Running regression cross-validation..."

        ):

            cv_r2_scores = cross_val_score(

                regression_model,

                X,

                y,

                cv=cv,

                scoring="r2"

            )


            cv_mae_scores = -cross_val_score(

                regression_model,

                X,

                y,

                cv=cv,

                scoring="neg_mean_absolute_error"

            )


        cv_r2_mean = (
            cv_r2_scores.mean()
        )


        cv_mae_mean = (
            cv_mae_scores.mean()
        )


        # ====================================================
        # CV SUMMARY
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(

                "5-Fold CV R²",

                f"{cv_r2_mean:.4f}"

            )


        with col2:

            st.metric(

                "5-Fold CV MAE",

                f"{cv_mae_mean:.4f}"

            )


        # ====================================================
        # CV TABLE
        # ====================================================

        cv_results = pd.DataFrame({

            "Fold": [
                1,
                2,
                3,
                4,
                5
            ],

            "R² Score":

                cv_r2_scores.round(4),

            "MAE":

                cv_mae_scores.round(4)

        })


        st.subheader(
            "📋 Individual Fold Results"
        )


        st.dataframe(

            cv_results,

            use_container_width=True

        )


        # ====================================================
        # CV GRAPH
        # ====================================================

        st.subheader(
            "📊 Regression Cross-Validation Performance"
        )


        fig, ax = plt.subplots()


        ax.plot(

            cv_results["Fold"],

            cv_results["R² Score"],

            marker="o",

            label="Linear Regression R²"

        )


        ax.set_xlabel(
            "Fold"
        )


        ax.set_ylabel(
            "R² Score"
        )


        ax.set_title(
            "5-Fold Cross-Validation R² Score"
        )


        ax.legend()


        ax.grid(
            True
        )


        st.pyplot(fig)


        plt.close(fig)


    else:

        cv_r2_mean = None

        cv_mae_mean = None


        st.warning(

            "⚠️ At least 5 samples are required "
            "for 5-fold cross-validation."

        )


    # ========================================================
    # ACTUAL VS PREDICTED TABLE
    # ========================================================

    st.header(
        "🔎 Actual vs Predicted Values"
    )


    prediction_results = pd.DataFrame({

        "Actual":

            y_test.values,

        "Predicted":

            regression_predictions

    })


    prediction_results["Error"] = (

        prediction_results["Actual"]

        -

        prediction_results["Predicted"]

    )


    st.dataframe(

        prediction_results.round(4),

        use_container_width=True

    )


    # ========================================================
    # ACTUAL VS PREDICTED GRAPH
    # ========================================================

    st.subheader(
        "📈 Actual vs Predicted Graph"
    )


    fig, ax = plt.subplots()


    ax.scatter(

        y_test,

        regression_predictions,

        alpha=0.7

    )


    min_value = min(

        y_test.min(),

        regression_predictions.min()

    )


    max_value = max(

        y_test.max(),

        regression_predictions.max()

    )


    ax.plot(

        [min_value, max_value],

        [min_value, max_value],

        linestyle="--",

        label="Perfect Prediction"

    )


    ax.set_xlabel(
        "Actual Values"
    )


    ax.set_ylabel(
        "Predicted Values"
    )


    ax.set_title(
        "Actual vs Predicted Values"
    )


    ax.legend()


    ax.grid(
        True
    )


    st.pyplot(fig)


    plt.close(fig)


    # ========================================================
    # REGRESSION MODEL COMPARISON
    # ========================================================

    st.header(
        "🏆 Regression Model Comparison"
    )


    regression_comparison = pd.DataFrame({

        "Model": [

            "Linear Regression"

        ],

        "MAE": [

            mae

        ],

        "MSE": [

            mse

        ],

        "RMSE": [

            rmse

        ],

        "R² Score": [

            r2

        ],

        "CV R²": [

            cv_r2_mean

            if cv_r2_mean is not None

            else np.nan

        ]

    })


    st.dataframe(

        regression_comparison.round(4),

        use_container_width=True

    )


    # ========================================================
    # REGRESSION PERFORMANCE CHART
    # ========================================================

    st.subheader(
        "📊 Regression Performance"
    )


    chart_data = pd.DataFrame({

        "Metric": [

            "R² Score",

            "CV R²"

        ],

        "Score": [

            r2,

            cv_r2_mean

            if cv_r2_mean is not None

            else np.nan

        ]

    })


    st.bar_chart(

        chart_data.set_index(
            "Metric"
        )

    )


    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    st.header(
        "🏆 Final Recommendation"
    )


    if r2 >= 0.70:

        st.success(

            f"🏆 Linear Regression performed well "
            f"with an R² Score of **{r2:.4f}**."

        )

    elif r2 >= 0.50:

        st.info(

            f"ℹ️ Linear Regression achieved a "
            f"moderate R² Score of **{r2:.4f}**."

        )

    else:

        st.warning(

            f"⚠️ Linear Regression achieved an "
            f"R² Score of **{r2:.4f}**. "
            f"Other regression algorithms may "
            f"perform better on this dataset."

        )


    # ========================================================
    # REGRESSION CONCLUSION
    # ========================================================

    st.header(
        "📝 Regression Conclusion"
    )


    st.write(

        f"The Linear Regression model was trained and "
        f"evaluated using MAE, MSE, RMSE and R² Score. "
        f"The model achieved an R² Score of **{r2:.4f}** "
        f"and an RMSE of **{rmse:.4f}**."

    )


    if cv_r2_mean is not None:

        st.write(

            f"During 5-fold cross-validation, the average "
            f"R² Score was **{cv_r2_mean:.4f}** and the "
            f"average MAE was **{cv_mae_mean:.4f}**."

        )


    # ========================================================
    # DOWNLOAD REGRESSION RESULTS
    # ========================================================

    st.header(
        "⬇️ Download Regression Results"
    )


    regression_csv = (

        regression_comparison

        .to_csv(
            index=False
        )

    )


    st.download_button(

        label="📥 Download Regression Results",

        data=regression_csv,

        file_name="regression_model_results.csv",

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