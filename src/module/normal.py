import numpy as np
import config as config
import module.model as model
import lightgbm as lgb

class IndependentForecastingStrategy:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_lat = {h: model.get_model(model_name) for h in config.HORIZONS}
        self.model_lon = {h: model.get_model(model_name) for h in config.HORIZONS}
        self.model_wind = {h: model.get_model(model_name) for h in config.HORIZONS}

    def train(self, X_train, y_train_lat_dict, y_train_lon_dict, y_train_wind_dict, X_val, y_val_lat_dict, y_val_lon_dict, y_val_wind_dict):
        for horizon in config.HORIZONS:
            y_tr_lat = y_train_lat_dict[horizon]
            y_tr_lon = y_train_lon_dict[horizon]
            y_tr_wind = y_train_wind_dict[horizon]
            y_v_lat = y_val_lat_dict[horizon]
            y_v_lon = y_val_lon_dict[horizon]
            y_v_wind = y_val_wind_dict[horizon]

            if self.model_name == 'lightgbm':
                self.model_lat[horizon].fit(
                    X_train, y_tr_lat, eval_set=[(X_val, y_v_lat)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                self.model_lon[horizon].fit(
                    X_train, y_tr_lon, eval_set=[(X_val, y_v_lon)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                self.model_wind[horizon].fit(
                    X_train, y_tr_wind, eval_set=[(X_val, y_v_wind)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            elif self.model_name == 'xgboost':
                self.model_lat[horizon].fit(X_train, y_tr_lat, eval_set=[(X_val, y_v_lat)], verbose=False)
                self.model_lon[horizon].fit(X_train, y_tr_lon, eval_set=[(X_val, y_v_lon)], verbose=False)
                self.model_wind[horizon].fit(X_train, y_tr_wind, eval_set=[(X_val, y_v_wind)], verbose=False)
            elif self.model_name == 'gdbt':
                self.model_lat[horizon].fit(X_train, y_tr_lat)
                self.model_lon[horizon].fit(X_train, y_tr_lon)
                self.model_wind[horizon].fit(X_train, y_tr_wind)
            else:
                print('Model is not in the list!')
        
    def predict(self, X_test):
        predictions_lat = {}
        predictions_lon = {}
        predictions_wind = {}
        for horizon in config.HORIZONS:
            predictions_lat[horizon] = self.model_lat[horizon].predict(X_test)
            predictions_lon[horizon] = self.model_lon[horizon].predict(X_test)
            predictions_wind[horizon] = self.model_wind[horizon].predict(X_test)
        return predictions_lat, predictions_lon, predictions_wind

class EnsembleForecastingStrategy:
    def __init__(self, models: list):
        self.models = models
        self.strategies = [IndependentForecastingStrategy(m) for m in models]

    def train(self, X_train, y_train_lat_dict, y_train_lon_dict, y_train_wind_dict, X_val, y_val_lat_dict, y_val_lon_dict, y_val_wind_dict):
        for strategy in self.strategies:
            print(f"   [Ensemble] Training {strategy.model_name}...")
            strategy.train(X_train, y_train_lat_dict, y_train_lon_dict, y_train_wind_dict, X_val, y_val_lat_dict, y_val_lon_dict, y_val_wind_dict)
            
    def predict(self, X_test):
        all_pred_lat = {h: [] for h in config.HORIZONS}
        all_pred_lon = {h: [] for h in config.HORIZONS}
        all_pred_wind = {h: [] for h in config.HORIZONS}
        
        for strategy in self.strategies:
            p_lat, p_lon, p_wind = strategy.predict(X_test)
            for h in config.HORIZONS:
                all_pred_lat[h].append(p_lat[h])
                all_pred_lon[h].append(p_lon[h])
                all_pred_wind[h].append(p_wind[h])
                
        final_lat = {h: np.mean(all_pred_lat[h], axis=0) for h in config.HORIZONS}
        final_lon = {h: np.mean(all_pred_lon[h], axis=0) for h in config.HORIZONS}
        final_wind = {h: np.mean(all_pred_wind[h], axis=0) for h in config.HORIZONS}
        
        return final_lat, final_lon, final_wind

class StackingForecastingStrategy:
    def __init__(self, models: list):
        self.models = models
        self.strategies = [IndependentForecastingStrategy(m) for m in models]
        from sklearn.linear_model import Ridge
        self.meta_lat = {h: Ridge() for h in config.HORIZONS}
        self.meta_lon = {h: Ridge() for h in config.HORIZONS}
        self.meta_wind = {h: Ridge() for h in config.HORIZONS}

    def train(self, X_train, y_train_lat_dict, y_train_lon_dict, y_train_wind_dict, X_val, y_val_lat_dict, y_val_lon_dict, y_val_wind_dict):
        # 1. Train base models
        for strategy in self.strategies:
            print(f"   [Stacking] Training base model {strategy.model_name}...")
            strategy.train(X_train, y_train_lat_dict, y_train_lon_dict, y_train_wind_dict, X_val, y_val_lat_dict, y_val_lon_dict, y_val_wind_dict)
            
        # 2. Predict on X_val to get features for meta model
        meta_features_lat = {h: [] for h in config.HORIZONS}
        meta_features_lon = {h: [] for h in config.HORIZONS}
        meta_features_wind = {h: [] for h in config.HORIZONS}
        
        for strategy in self.strategies:
            p_lat, p_lon, p_wind = strategy.predict(X_val)
            for h in config.HORIZONS:
                meta_features_lat[h].append(p_lat[h])
                meta_features_lon[h].append(p_lon[h])
                meta_features_wind[h].append(p_wind[h])
                
        # 3. Train meta models
        print("   [Stacking] Training Meta-model (Ridge)...")
        for h in config.HORIZONS:
            X_meta_lat = np.column_stack(meta_features_lat[h])
            X_meta_lon = np.column_stack(meta_features_lon[h])
            X_meta_wind = np.column_stack(meta_features_wind[h])
            
            self.meta_lat[h].fit(X_meta_lat, y_val_lat_dict[h])
            self.meta_lon[h].fit(X_meta_lon, y_val_lon_dict[h])
            self.meta_wind[h].fit(X_meta_wind, y_val_wind_dict[h])
            
    def predict(self, X_test):
        meta_features_lat = {h: [] for h in config.HORIZONS}
        meta_features_lon = {h: [] for h in config.HORIZONS}
        meta_features_wind = {h: [] for h in config.HORIZONS}
        
        for strategy in self.strategies:
            p_lat, p_lon, p_wind = strategy.predict(X_test)
            for h in config.HORIZONS:
                meta_features_lat[h].append(p_lat[h])
                meta_features_lon[h].append(p_lon[h])
                meta_features_wind[h].append(p_wind[h])
                
        final_lat = {}
        final_lon = {}
        final_wind = {}
        
        for h in config.HORIZONS:
            X_meta_lat = np.column_stack(meta_features_lat[h])
            X_meta_lon = np.column_stack(meta_features_lon[h])
            X_meta_wind = np.column_stack(meta_features_wind[h])
            
            final_lat[h] = self.meta_lat[h].predict(X_meta_lat)
            final_lon[h] = self.meta_lon[h].predict(X_meta_lon)
            final_wind[h] = self.meta_wind[h].predict(X_meta_wind)
            
        return final_lat, final_lon, final_wind
