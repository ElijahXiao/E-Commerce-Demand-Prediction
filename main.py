import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from data_preprocess import data_preprocessing
from sklearn.preprocessing import MinMaxScaler

def main():
    # Load the data
    df = pd.read_csv('Sales Transaction v.4a.csv')
    f = open('84879.txt', 'a')
    df = data_preprocessing(df)
        
    # Split the data into training and testing sets
    y = pd.DataFrame(df['84879'], columns=['84879'])
    X = df.drop(['84879'], axis=1)
    # print(X)
    # print(y)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    num_test = 8
    X_train, y_train = X[:-num_test], y[:-num_test]
    X_test, y_test = X[-num_test:], y[-num_test:]
    
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Create and fit the MinMaxScaler for the input features
    X_scaler = MinMaxScaler()
    X_train = pd.DataFrame(X_scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(X_scaler.fit_transform(X_test), columns=X_test.columns)

    # Create and fit the MinMaxScaler for the target variable
    y_scaler = MinMaxScaler()
    y_train = pd.DataFrame(y_scaler.fit_transform(y_train), columns=y_train.columns)
    y_test = pd.DataFrame(y_scaler.fit_transform(y_test), columns=y_test.columns)
      
    # Get the number of training examples
    num_train = X_train.shape[0]

    # Generate a list of shuffled indices for the training data
    shuffled_indices = list(range(num_train))
    np.random.shuffle(shuffled_indices)
    # print("shuffled_indices:\n", shuffled_indices)

    # Shuffle the training data and labels using the shuffled indices
    X_train = X_train.iloc[shuffled_indices, :]
    y_train = y_train.iloc[shuffled_indices]
    
    # print("X train\n",X_train)
    # print("X test\n",X_test)
    # print("y train\n",y_train)
    # print("t test\n",y_test)
    
    y_train = y_train.values.ravel()
     
    # Train a Random Forest Regressor model
    param_grid_rf = {
      'n_estimators': [*range(1,50,2)],
      'max_depth': [*range(5,20,2)]
    }
    
    # rf_model = RandomForestRegressor()
    # grid_search_rf = GridSearchCV(
    #   estimator=rf_model,
    #   param_grid=param_grid_rf,
    #   scoring='neg_mean_squared_error',
    #   cv=5,
    #   n_jobs=-1,
    #   verbose=True
    # )
    # param_grid_rf = {
    #                  'n_estimators': [*range(20,400,10)]
    #                  , 'max_depth': [*range(5,25,2)]
                     
                    
    #                 }
    
    # grid_search_rf.fit(X_train, y_train)
    # print('Best Params:', grid_search_rf.best_params_)
    # print('Best Score:', grid_search_rf.best_score_)
    # return
    rf_model = RandomForestRegressor(n_estimators=400, max_depth=7)
    rf_model.fit(X_train, y_train)
    
    # Train a SVR model
    svr_model = SVR()
    param_grid_svr = {'kernel': ['linear','poly','rbf','sigmoid'],
                      'C': [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],
                      'degree': [3,8],
                      'coef0': [0.01,0.1,0.5],
                      'gamma': ('auto','scale'),
                      'tol': [1e-3, 1e-4, 1e-5, 1e-6]}
    grid_search_svr = GridSearchCV(
      estimator=svr_model,
      param_grid=param_grid_svr,
      scoring='neg_mean_squared_error',
      cv=5,
      n_jobs=-1,
      verbose=True
    )
    # grid_search_svr.fit(X_train, y_train)
    # print('Best Params:', grid_search_svr.best_params_)
    # print('Best Score:', grid_search_svr.best_score_)
      
    svr_model = SVR(C=3.5, gamma=0.03)
    svr_model.fit(X_train, y_train)
    
    # Train a Gradient Boosting Regressor model
    gbrt_model = GradientBoostingRegressor(n_estimators=200, max_depth=5)
    gbrt_model.fit(X_train, y_train)
    
    # Evaluate the performance of the models on the testing set
    rf_y_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_y_pred)
    
    svr_y_pred = svr_model.predict(X_test)
    svr_r2 = r2_score(y_test, svr_y_pred)
    
    gbrt_y_pred = gbrt_model.predict(X_test)
    gbrt_r2 = r2_score(y_test, gbrt_y_pred)
    
    
    # ensemble learning 
    rf_x_pred = rf_model.predict(X_train)
    svr_x_pred = svr_model.predict(X_train)
    gbrt_x_pred = gbrt_model.predict(X_train)
    stacking_X_train = np.column_stack((rf_x_pred, gbrt_x_pred, svr_x_pred))
    el_model = LinearRegression()
    el_model.fit(stacking_X_train, y_train)
    
    stacking_X_test = np.column_stack((rf_y_pred, gbrt_y_pred, svr_y_pred))    
    el_y_pred = el_model.predict(stacking_X_test)
    el_r2 = r2_score(y_test, el_y_pred)
    
    
    f.write(f"Random Forest R2 score: {rf_r2}\n")
    f.write(f"SVR R2 score: {svr_r2}\n")
    f.write(f"Gradient Boosting R2 score: {gbrt_r2}\n")
    f.write(f"Ensemble Learning R2 score: {el_r2}\n")
    
    # reverse normalization
    y = y_scaler.inverse_transform(y_test['84879'].to_numpy().reshape(-1, 1)).ravel()
    y_test['84879'] = y
    
    rf_y_pred = y_scaler.inverse_transform(rf_y_pred.reshape(-1, 1)).ravel()
    svr_y_pred = y_scaler.inverse_transform(svr_y_pred.reshape(-1, 1)).ravel()
    gbrt_y_pred = y_scaler.inverse_transform(gbrt_y_pred.reshape(-1, 1)).ravel()
    el_y_pred = y_scaler.inverse_transform(el_y_pred.reshape(-1, 1)).ravel()
 
    # absolute difference between predicted and target values    
    diff_rf = abs(y_test['84879'] - pd.Series(rf_y_pred))
    mae_rf =np.mean(diff_rf)
    mape_rf = np.mean(diff_rf / y_test['84879']) * 100
    
    diff_svr = abs(y_test['84879'] - pd.Series(svr_y_pred))
    mae_svr =np.mean(diff_svr)
    mape_svr = np.mean(diff_svr / y_test['84879']) * 100
    
    diff_gbrt = abs(y_test['84879'] - pd.Series(gbrt_y_pred))
    mae_gbrt =np.mean(diff_gbrt)
    mape_gbrt = np.mean(diff_gbrt / y_test['84879']) * 100
    
    diff_el = abs(y_test['84879'] - pd.Series(el_y_pred))
    mae_el =np.mean(diff_el)
    mape_el = np.mean(diff_el / y_test['84879']) * 100
    
    f.write('\nPredicted values of 84879 in next 8 weeks\n')
    f.write(f'{el_y_pred}')      
    f.write("\nAbsolute difference between predicted and target values")
    f.write(f"\nRandom Forest: {diff_rf}")
    f.write(f"\nSVR: {diff_svr}")
    f.write(f"\nGradient Boosting: {diff_gbrt}")
    f.write(f"\nEnsemble Learning: {diff_el}")
    f.write("\nMean Absolute Error between predicted and target values")
    f.write(f"\nRandom Forest: {mae_rf}")
    f.write(f"\nSVR: {mae_svr}")
    f.write(f"\nGradient Boosting: {mae_gbrt}")
    f.write(f"\nEnsemble Learning: {mae_el}")
    f.write("\nMean Absolute Percentage Error between predicted and target values")
    f.write(f"\nRandom Forest: {mape_rf}")
    f.write(f"\nSVR: {mape_svr}")
    f.write(f"\nGradient Boosting: {mape_gbrt}")
    f.write(f"\nEnsemble Learning: {mape_el}")
    f.close()
    
    # Create plots of predicted values vs target values for each model
    # Plot y_test and rf_y_pred on the same graph
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(y_test)+1), y_test, label='Target Value')
    plt.plot(range(1, len(y_test)+1), rf_y_pred, label='Predicted Value')
    plt.xlabel('Time')
    plt.ylabel('Commodity demand')
    plt.title('Random Forest')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(y_test)+1), y_test, label='Target Value')
    plt.plot(range(1, len(y_test)+1), svr_y_pred, label='Predicted Value')
    plt.xlabel('Time')
    plt.ylabel('Commodity demand')
    plt.title('SVR')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(y_test)+1), y_test, label='Target Value')
    plt.plot(range(1, len(y_test)+1), gbrt_y_pred, label='Predicted Value')
    plt.xlabel('Time')
    plt.ylabel('Commodity demand')
    plt.title('Gradient Boosting')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(y_test)+1), y_test, label='Target Value')
    plt.plot(range(1, len(y_test)+1), el_y_pred, label='Predicted Value')
    plt.xlabel('Time')
    plt.ylabel('Commodity demand')
    plt.title('Ensemble Learning')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
  main()