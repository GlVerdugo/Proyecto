"""Main module."""
from load.load_data import DataRetriever
from train.train_data import HeartDiseasePipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score

DATASETS_DIR = './datasets/'
URL = pd.read_csv('C:/Users/glverdugo/Documents/Maestria/MLops/Proyecto/heart_disease_prediction/heart_disease_prediction/data/heart_2020_cleaned.csv')
RETRIEVED_DATA = 'heart_2020_cleaned.csv'


SEED_SPLIT = 44
TRAIN_DATA_FILE = DATASETS_DIR + 'train.csv'
TEST_DATA_FILE  = DATASETS_DIR + 'test.csv'


TARGET = 'HeartDisease'
FEATURES = ['BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex',
            'AgeCategory', 'Race','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma',
            'KidneyDisease','SkinCancer' ]

NUMERICAL_VARS = ['BMI','PhysicalHealth','MentalHealth','SleepTime']
CATEGORICAL_VARS = ['Sex','AgeCategory','Race','GenHealth']
BINARI_VARS = ['Smoking','AlcoholDrinking','Stroke','DiffWalking','Diabetic',
                    'PhysicalActivity','Asthma','KidneyDisease','SkinCancer' ]


SEED_MODEL = 44

SELECTED_FEATURES = ['BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex',
            'AgeCategory', 'Race','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma',
            'KidneyDisease','SkinCancer' ]

TRAINED_MODEL_DIR = './models/'
PIPELINE_NAME = 'KNeighbors_Classifier'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'


if __name__ == "__main__":
    filepath = "heart_disease_prediction\heart_disease_prediction\data\heart_2020_cleaned.csv"
    data_retriever = DataRetriever(filepath)
    df = data_retriever.read_data()
    print(df)
    
    # Instantiate the HeartDiseasePipeline class
    HeartDisease_data_pipeline = HeartDiseasePipeline(seed_model=SEED_MODEL,
                                                numerical_vars=NUMERICAL_VARS, 
                                                categorical_vars=CATEGORICAL_VARS,
                                                binari_vars=BINARI_VARS,
                                                features=FEATURES,
                                                selected_features=SELECTED_FEATURES)
    
    # Read data
    df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
                                                        df.drop(TARGET, axis=1),
                                                        df[TARGET],
                                                        test_size=0.2,
                                                        random_state=404
                                                   )
    
    
    KNeighborsClassifier_model = HeartDisease_data_pipeline.fit_KNeighborsClassifier(X_train, y_train)
    
    X_test = HeartDisease_data_pipeline.PIPELINE.fit_transform(X_test)
    y_pred = KNeighborsClassifier_model.predict(X_test)
    
    class_pred = KNeighborsClassifier_model.predict(X_test)
    proba_pred = KNeighborsClassifier_model.predict_proba(X_test)[:,1]
    print(f'test roc-auc : {roc_auc_score(y_test, proba_pred)}')
    print(f'test accuracy: {accuracy_score(y_test, class_pred)}')
    
    # # Save the model using joblib
    save_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE
    joblib.dump(KNeighborsClassifier_model, save_path)
    print(f"Model saved in {save_path}")
    
