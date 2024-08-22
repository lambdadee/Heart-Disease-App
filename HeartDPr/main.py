import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
def get_clean_data():
    data = pd.read_csv("HeartDPr/Heart_Disease_Prediction.csv")
    data["Heart Disease"] = data["Heart Disease"].map({"Absence" : 0, "Presence" : 1})
    print(data.info())
    return data

def create_model(data):

    x = data.drop(["Heart Disease"], axis = 1)
    y = data["Heart Disease"]

    scaler = StandardScaler()

    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Model score: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler





def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("model/data.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()