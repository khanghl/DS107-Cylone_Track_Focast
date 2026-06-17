import config as config
import models.model as model
import lightgbm as lgb

class IndependentForecastingStrategy:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_lat = {h: model.get_model(model_name) for h in config.HORIZONS}
        self.model_lon = {h: model.get_model(model_name) for h in config.HORIZONS}

    def train(self, X_train, y_train_lat_dict, y_train_lon_dict, X_val, y_val_lat_dict, y_val_lon_dict):
        for horizon in config.HORIZONS:
            y_tr_lat = y_train_lat_dict[horizon]
            y_tr_lon = y_train_lon_dict[horizon]
            y_v_lat = y_val_lat_dict[horizon]
            y_v_lon = y_val_lon_dict[horizon]

            if self.model_name == 'lightgbm':
                self.model_lat[horizon].fit(
                    X_train, y_tr_lat, eval_set=[(X_val, y_v_lat)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                self.model_lon[horizon].fit(
                    X_train, y_tr_lon, eval_set=[(X_val, y_v_lon)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            elif self.model_name == 'xgboost':
                self.model_lat[horizon].fit(X_train, y_tr_lat, eval_set=[(X_val, y_v_lat)], verbose=False)
                self.model_lon[horizon].fit(X_train, y_tr_lon, eval_set=[(X_val, y_v_lon)], verbose=False)
            elif self.model_name == 'gdbt':
                self.model_lat[horizon].fit(X_train, y_tr_lat)
                self.model_lon[horizon].fit(X_train, y_tr_lon)
            else:
                print('Model is not in the list!')
        
    def predict(self, X_test):
        predictions_lat = {}
        predictions_lon = {}
        for horizon in config.HORIZONS:
            predictions_lat[horizon] = self.model_lat[horizon].predict(X_test)
            predictions_lon[horizon] = self.model_lon[horizon].predict(X_test)
        return predictions_lat, predictions_lon