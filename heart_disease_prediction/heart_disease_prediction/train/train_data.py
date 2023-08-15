from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from preprocess.preprocess_data import (
    MissingIndicator,
    CategoricalImputer,
    NumericalImputer,
    OneHotEncoder,
    TransformerBinari,
    ColumTransformer
    )

class HeartDiseasePipeline:
    """
    A class representing the Heart Disease data processing and modeling pipeline.

    Attributes:
        NUMERICAL_VARS (list): A list of numerical variables in the dataset.
        CATEGORICAL_VARS (list): A list of categorical variables in the dataset.
        SEED_MODEL (int): A seed value for reproducibility.

    Methods:
        create_pipeline(): Create and return the Heart Disease data processing pipeline.
    """
    
    def __init__(self, seed_model, numerical_vars,binari_vars, categorical_vars, selected_features, features):
        self.SEED_MODEL = seed_model
        self.NUMERICAL_VARS = numerical_vars
        self.CATEGORICAL_VARS = categorical_vars
        self.BINARI_VARS= binari_vars
        self.SEED_MODEL = seed_model
        self.SELECTED_FEATURES = selected_features
        self.FEATURES=features
        
        
    def create_pipeline(self):
        """
        Create and return the Heart Disease data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.PIPELINE = Pipeline(
            [
                                ('missing_indicator', MissingIndicator(variables=self.NUMERICAL_VARS)),
                                ('binari', TransformerBinari(variables=self.BINARI_VARS)),
                                ('colum_transformer', ColumTransformer(variables=self.CATEGORICAL_VARS)),
                                ('dummy_vars', OneHotEncoder(variables=self.CATEGORICAL_VARS)),
                                ('scaling', MinMaxScaler()),

                              ]
        )
        return self.PIPELINE
    
    def fit_KNeighborsClassifier (self, X_train, y_train):
        """
        Fit a Neighbors Classifier model using the predefined data preprocessing pipeline.

        Parameters:
        - X_train (pandas.DataFrame or numpy.ndarray): The training input data.
        - y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
        - logistic_KNeighborsClassifier  (NeighborsClassifier): The fitted  classifications model based on data neighbors.
        """
        KNClassifier= KNeighborsClassifier(n_neighbors = 5)
        pipeline = self.create_pipeline()
        pipeline.fit(X_train, y_train)
        KNClassifier.fit(pipeline.transform(X_train), y_train)
        return KNClassifier
    
    def transform_test_data(self, X_test):
        """
        Apply the data preprocessing pipeline on the test data.

        Parameters:
        - X_test (pandas.DataFrame or numpy.ndarray): The test input data.

        Returns:
        - transformed_data (pandas.DataFrame or numpy.ndarray): The preprocessed test data.
        """
        pipeline = self.create_pipeline()
        return pipeline.transform(X_test)