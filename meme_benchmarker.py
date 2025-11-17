#!/usr/bin/env python3

import time
import random
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class MemeBenchmarker:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.models = {}
        self.feature_names = []
        self.meme_data = []
        self.model_results = {}
        
        self.models = {
            'lifespan': RandomForestRegressor(n_estimators=100, random_state=42),
            'peak_timing': GradientBoostingRegressor(random_state=42),
            'decay_rate': LinearRegression(),
            'viral_potential': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'sustainability': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        print("initialized")
    
    def get_trends_data(self, meme_name, timeframe='2022-01-01 2024-12-31'):
        try:
            
            self.pytrends.build_payload([meme_name], cat=0, timeframe=timeframe, geo='', gprop='')
            interest_data = self.pytrends.interest_over_time()
            
            if interest_data.empty:
                print(f"no data: {meme_name}")
                return None
                
            return interest_data
            
        except Exception as e:
            print(f"error: {str(e)}")
            return None
    
    def extract_comprehensive_features(self, trends_df, meme_name):
        if trends_df is None or trends_df.empty:
            return None
            
        interest_values = trends_df[meme_name].values
        
        features = {
            'meme_name': meme_name,
            'max_interest': np.max(interest_values),
            'mean_interest': np.mean(interest_values),
            'median_interest': np.median(interest_values),
            'std_interest': np.std(interest_values),
            'variance': np.var(interest_values),
            'total_points': len(interest_values),
            'non_zero_points': np.count_nonzero(interest_values),
            'zero_ratio': np.sum(interest_values == 0) / len(interest_values)
        }
        
        peak_idx = np.argmax(interest_values)
        features['peak_position'] = peak_idx / len(interest_values)
        features['peak_value'] = interest_values[peak_idx]
        
        if peak_idx < len(interest_values) - 1:
            post_peak = interest_values[peak_idx:]
            features['decay_rate'] = self._calculate_decay_rate(post_peak)
            features['sustained_interest'] = np.mean(post_peak)
        else:
            features['decay_rate'] = 0
            features['sustained_interest'] = 0
        
        if peak_idx > 0:
            pre_peak = interest_values[:peak_idx]
            features['growth_rate'] = self._calculate_growth_rate(pre_peak)
        else:
            features['growth_rate'] = 0
        
        features['skewness'] = self._calculate_skewness(interest_values)
        features['kurtosis'] = self._calculate_kurtosis(interest_values)
        
        features['volatility'] = np.std(np.diff(interest_values))
        features['max_change'] = np.max(np.abs(np.diff(interest_values)))
        
        features['trend_persistence'] = self._calculate_trend_persistence(interest_values)
        
        features['lifespan_days'] = self._calculate_lifespan(interest_values)
        features['peak_timing_ratio'] = features['peak_position']
        features['decay_rate_target'] = features['decay_rate']
        features['viral_score'] = self._calculate_viral_score(interest_values)
        features['sustainability_score'] = self._calculate_sustainability_score(interest_values)
        
        return features
    
    def _calculate_decay_rate(self, values):
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        if np.sum(x**2) == 0:
            return 0
        slope = np.polyfit(x, values, 1)[0]
        return abs(slope)
    
    def _calculate_growth_rate(self, values):
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        if np.sum(x**2) == 0:
            return 0
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def _calculate_skewness(self, values):
        if len(values) < 3:
            return 0
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return 0
        return np.mean(((values - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, values):
        if len(values) < 4:
            return 0
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return 0
        return np.mean(((values - mean_val) / std_val) ** 4) - 3
    
    def _calculate_trend_persistence(self, values):
        if len(values) < 2:
            return 0
        changes = np.diff(values)
        positive_changes = np.sum(changes > 0)
        total_changes = len(changes)
        if total_changes == 0:
            return 0
        return positive_changes / total_changes
    
    def _calculate_lifespan(self, values):
        max_interest = np.max(values)
        peak_idx = np.argmax(values)
        
        rise_threshold = max_interest * 0.2
        first_active = peak_idx
        for i in range(peak_idx + 1):
            if values[i] >= rise_threshold:
                first_active = i
                break
        
        death_threshold = max_interest * 0.25
        consecutive_low = 0
        last_active = peak_idx
        
        for i in range(peak_idx, len(values)):
            if values[i] >= death_threshold:
                last_active = i
                consecutive_low = 0
            else:
                consecutive_low += 1
                if consecutive_low >= 6:
                    last_active = i - 6
                    break
                last_active = i
        
        span_weeks = max(1, last_active - first_active + 1)
        span_days = span_weeks * 7
        
        return span_days
    
    def _calculate_viral_score(self, values):
        max_interest = np.max(values)
        mean_interest = np.mean(values)
        
        peak_idx = np.argmax(values)
        peak_position = peak_idx / len(values) if len(values) > 0 else 0.5
        
        if peak_idx > 0:
            pre_peak = values[:peak_idx]
            if len(pre_peak) > 1:
                growth_rate = np.mean(np.diff(pre_peak))
            else:
                growth_rate = 0
        else:
            growth_rate = 0
        
        if peak_idx > 0:
            velocity = max_interest / (peak_idx + 1)
        else:
            velocity = max_interest
        
        top_quartile_threshold = np.percentile(values, 75)
        momentum = np.sum(values > top_quartile_threshold) / len(values) if len(values) > 0 else 0
        
        max_interest_norm = max_interest / 100.0
        peak_early_factor = max(0, 1 - peak_position * 2)
        growth_factor = max(0, min(1, (growth_rate + 5) / 10))
        velocity_factor = min(1, velocity / 10)
        momentum_factor = momentum
        
        viral_score = (
            0.3 * max_interest_norm +
            0.25 * peak_early_factor +
            0.2 * growth_factor +
            0.15 * velocity_factor +
            0.1 * momentum_factor
        )
        
        return min(max(viral_score, 0.0), 1.0)
    
    def _calculate_sustainability_score(self, values):
        mean_interest = np.mean(values)
        sustained_interest = np.mean(values[len(values)//2:])
        decay_rate = self._calculate_decay_rate(values)
        
        sustainability = (mean_interest / 100) * (sustained_interest / 100) * (1 - decay_rate)
        return min(sustainability, 1.0)
    
    def create_meme_dataset(self, meme_list=None):
        if meme_list is None:
            meme_list = [
                "distracted boyfriend", "woman yelling at cat", "this is fine", "drake pointing",
                "change my mind", "expanding brain", "galaxy brain", "stonks", "big chungus",
                "wojak", "pepe", "doge", "crying laughing emoji", "skull emoji", "capybara",
                "sigma grindset", "main character", "NPC", "chad", "virgin", "gigachad",
                "soyjak", "doomer", "boomer", "zoomer", "ok boomer", "no cap", "bet",
                "fr fr", "on god", "bussin", "slay", "periodt", "and I oop", "sksksk",
                "vsco girl", "e-girl", "simps", "thirst trap", "main character energy",
                "pick me girl", "not like other girls", "basic", "extra", "cringe",
                "based", "redpilled", "bluepilled", "woke", "cancel culture", "Karen",
                "millennial", "gen z", "gen alpha", "cheugy", "stan", "ship", "OTP",
                "BFF", "FOMO", "YOLO", "main character syndrome", "pick me energy",
                "hot girl summer", "side character", "background character", "NPC energy",
                "protagonist energy", "antagonist energy", "villain arc", "redemption arc",
                "character development", "plot twist", "cliffhanger", "season finale",
                "series finale", "spin-off", "reboot", "remake", "sequel", "prequel",
                "rizz", "skibidi toilet", "ohio", "fanum tax", "sigma", "alpha",
                "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
                "rho", "tau", "upsilon", "phi", "chi", "psi", "omega"
            ]
        
        print(f"creating dataset: {len(meme_list)} memes")
        
        for i, meme in enumerate(meme_list):
            print(f"{i+1}/{len(meme_list)}: {meme}")
            
            trends_data = self.get_trends_data(meme)
            
            if trends_data is not None:
                features = self.extract_comprehensive_features(trends_data, meme)
                
                if features:
                    self.meme_data.append(features)
                    print(f"{meme}: {features['lifespan_days']}d, viral: {features['viral_score']:.2f}")
                else:
                    print(f"{meme}: no features")
            else:
                print(f"{meme}: no data")
            
            time.sleep(random.uniform(2, 4))
        
        print(f"done: {len(self.meme_data)} memes")
        return self.meme_data
    
    def train_ensemble_models(self, verbose=True):
        if len(self.meme_data) < 10:
            if verbose:
                print("not enough data")
            return False
        
        if verbose:
            print("training models")
        
        df = pd.DataFrame(self.meme_data)
        
        exclude_cols = ['meme_name', 'lifespan_days', 'peak_timing_ratio', 
                       'decay_rate_target', 'viral_score', 'sustainability_score']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_names]
        
        model_results = {}
        
        for model_name, model in self.models.items():
            if verbose:
                print(f"training {model_name}")
            
            if model_name == 'lifespan':
                y = df['lifespan_days']
            elif model_name == 'peak_timing':
                y = df['peak_timing_ratio']
            elif model_name == 'decay_rate':
                y = df['decay_rate_target']
            elif model_name == 'viral_potential':
                y = df['viral_score']
            elif model_name == 'sustainability':
                y = df['sustainability_score']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_name == 'viral_potential':
                mask_train = np.isfinite(y_train.values)
                mask_test = np.isfinite(y_test.values)
                X_train = X_train[mask_train]
                y_train = y_train[mask_train]
                X_test = X_test[mask_test]
                y_test = y_test[mask_test]
            
            try:
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                if model_name == 'viral_potential':
                    y_pred = np.clip(y_pred, 0, 1)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                mean_actual = np.mean(y_test.values)
                mape = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-8))) * 100
                
                model_results[model_name] = {
                    'mae': mae, 
                    'r2': r2, 
                    'rmse': rmse,
                    'mape': mape,
                    'test_size': len(y_test)
                }
                if verbose:
                    print(f"{model_name}: mae={mae:.3f}, r2={r2:.3f}")
            except Exception as e:
                if verbose:
                    print(f"{model_name}: error - {str(e)}")
                model_results[model_name] = {'mae': float('inf'), 'r2': -999, 'rmse': float('inf'), 'mape': float('inf'), 'test_size': 0}
        
        self.model_results = model_results
        if verbose:
            print("models trained")
        return True
    
    def predict_meme_performance(self, meme_name):
        if not all(model is not None for model in self.models.values()):
            print("not trained")
            return None
        
        print(f"predicting {meme_name}")
        
        trends_data = self.get_trends_data(meme_name)
        
        features = self.extract_comprehensive_features(trends_data, meme_name)
        
        feature_values = [features[col] for col in self.feature_names]
        X = np.array(feature_values).reshape(1, -1)
        
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(X)[0]
            if model_name == 'viral_potential':
                pred = np.clip(pred, 0.0, 1.0)
            predictions[model_name] = pred
        
        result = {
            'meme_name': meme_name,
            'predictions': predictions,
            'features': features,
            'trends_data': trends_data,
            'summary': self._create_prediction_summary(predictions)
        }
        
        return result
    
    def _create_prediction_summary(self, predictions):
        lifespan = predictions['lifespan']
        viral = np.clip(predictions['viral_potential'], 0.0, 1.0)
        sustainability = np.clip(predictions['sustainability'], 0.0, 1.0)
        peak_timing = np.clip(predictions['peak_timing'], 0.0, 1.0)
        decay = max(0, predictions['decay_rate'])
        
        summary = {
            'lifespan_days': round(lifespan, 1),
            'lifespan_weeks': round(lifespan / 7, 1),
            'viral_potential': f"{viral:.1%}",
            'sustainability': f"{sustainability:.1%}",
            'peak_timing': f"{peak_timing:.1%} through timeline",
            'decay_rate': f"{decay:.3f}",
            'overall_score': round((viral + sustainability) / 2, 2)
        }
        
        return summary
    
    def compare_memes(self, meme1, meme2):
        print(f"{meme1} vs {meme2}")
        
        pred1 = self.predict_meme_performance(meme1)
        pred2 = self.predict_meme_performance(meme2)
        
        if pred1 is None or pred2 is None:
            print("prediction failed")
            return None
        
        comparison = {
            'meme1': {'name': meme1, 'predictions': pred1['predictions'], 'summary': pred1['summary']},
            'meme2': {'name': meme2, 'predictions': pred2['predictions'], 'summary': pred2['summary']},
            'winner': self._determine_winner(pred1['predictions'], pred2['predictions'])
        }
        
        return comparison
    
    def _determine_winner(self, pred1, pred2):
        categories = {
            'lifespan': 'Longer lasting',
            'viral_potential': 'More viral',
            'sustainability': 'More sustainable',
            'peak_timing': 'Peaks earlier' if pred1['peak_timing'] < pred2['peak_timing'] else 'Peaks later',
            'decay_rate': 'Decays slower' if pred1['decay_rate'] < pred2['decay_rate'] else 'Decays faster'
        }
        
        winners = {}
        for category, description in categories.items():
            if pred1[category] > pred2[category]:
                winners[category] = 'meme1'
            else:
                winners[category] = 'meme2'
        
        overall1 = (pred1['viral_potential'] + pred1['sustainability']) / 2
        overall2 = (pred2['viral_potential'] + pred2['sustainability']) / 2
        
        winners['overall'] = 'meme1' if overall1 > overall2 else 'meme2'
        
        return winners
    
    def analyze_model_accuracy(self):
        if not self.model_results:
            print("no results - train models first")
            return
        
        print("\nmodel accuracy:")
        print("-" * 60)
        
        for model_name, results in self.model_results.items():
            if results['r2'] < -100:
                continue
                
            print(f"\n{model_name}:")
            print(f"  r2 score:     {results['r2']:.4f}")
            print(f"  mae:          {results['mae']:.4f}")
            print(f"  rmse:         {results['rmse']:.4f}")
            print(f"  mape:         {results['mape']:.2f}%")
            print(f"  test samples: {results['test_size']}")
        
        print("\n" + "-" * 60)
        
        avg_r2 = np.mean([r['r2'] for r in self.model_results.values() if r['r2'] > -100])
        avg_mae = np.mean([r['mae'] for r in self.model_results.values() if r['r2'] > -100])
        print(f"average r2: {avg_r2:.4f}")
        print(f"average mae: {avg_mae:.4f}")
        
        if len(self.meme_data) > 0:
            print(f"\ndataset size: {len(self.meme_data)} memes")
            print(f"features: {len(self.feature_names)}")
    
    def save_models(self, filename='meme_benchmarker_models.pkl'):
        model_data = {
            'models': self.models,
            'feature_names': self.feature_names,
            'meme_data': self.meme_data,
            'model_results': self.model_results
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"saved: {filename}")
        return True
    
    def load_models(self, filename='meme_benchmarker_models.pkl'):
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.feature_names = model_data['feature_names']
            self.meme_data = model_data['meme_data']
            self.model_results = model_data.get('model_results', {})
            
            print(f"loaded: {filename}")
            return True
            
        except Exception as e:
            print(f"load error: {str(e)}")
            return False

def main():
    print("meme benchmarker")
    
    benchmarker = MemeBenchmarker()
    
    if not benchmarker.load_models():
        print("creating models")
        
        benchmarker.create_meme_dataset()
        
        if benchmarker.train_ensemble_models():
            benchmarker.save_models()
        else:
            print("training failed")
            return
    
    while True:
        print("\n1. predict meme")
        print("2. compare memes")
        print("3. analyze")
        print("4. retrain")
        print("5. info")
        print("6. quit")
        
        choice = input("\nchoice: ").strip()
        
        if choice == '1':
            meme = input("Enter meme name: ").strip()
            if meme:
                result = benchmarker.predict_meme_performance(meme)
                if result:
                    print(f"\n{meme}:")
                    for key, value in result['summary'].items():
                        print(f"   {key}: {value}")
        
        elif choice == '2':
            meme1 = input("Enter first meme: ").strip()
            meme2 = input("Enter second meme: ").strip()
            if meme1 and meme2:
                comparison = benchmarker.compare_memes(meme1, meme2)
                if comparison:
                    print(f"\n{meme1} vs {meme2}")
                    print(f"winner: {comparison['winner']['overall']}")
                    for category, winner in comparison['winner'].items():
                        if category != 'overall':
                            print(f"{category}: {winner}")
        
        elif choice == '3':
            benchmarker.analyze_model_accuracy()
        
        elif choice == '4':
            print("retraining...")
            benchmarker.create_meme_dataset()
            if benchmarker.train_ensemble_models(verbose=False):
                benchmarker.save_models()
                print("retrained")
        
        elif choice == '5':
            print(f"models: {len(benchmarker.models)}")
            print(f"features: {len(benchmarker.feature_names)}")
            print(f"samples: {len(benchmarker.meme_data)}")
        
        elif choice == '6':
            print("bye")
            break
        
        else:
            print("invalid")

if __name__ == "__main__":
    main()
