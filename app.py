import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

# Set memory growth for TensorFlow to avoid GPU memory issues
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
except:
    pass

class ResourceAnalyzer:
    def __init__(self):
        self.resource_data = None
        self.cost_data = None
        self.workload_data = None
        self.scalers = {}
        self.models = {}
        
    def load_data(self):
        try:
            try:
                self.resource_data = pd.read_csv('cloud_resource_allocation_dataset.csv')
                self.cost_data = pd.read_csv('cloud_cost_optimization.csv')
                self.workload_data = pd.read_csv('cloud_workload_forecasting.csv')
            except Exception as e:
                # Create sample data if files don't exist
                self._create_sample_data()
            
            # Use only a subset of data for faster processing
            # Use only first 200 rows for faster processing
            self.workload_data = self.workload_data.iloc[:200]
            
            # Convert timestamp to datetime in workload data
            self.workload_data['timestamp'] = pd.to_datetime(self.workload_data['timestamp'])
            
        except Exception as e:
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data if no data is available"""
        
        # Create sample resource allocation data
        self.resource_data = pd.DataFrame({
            'initial_resource_pool': np.random.uniform(50, 150, 1000),
            'resource_utilization': np.random.uniform(0.4, 0.9, 1000),
            'allocation_efficiency': np.random.uniform(0.6, 0.95, 1000)
        })
        
        # Create sample cost data
        self.cost_data = pd.DataFrame({
            'cost_per_unit': np.random.uniform(0.05, 0.15, 1000),
            'total_cost': np.random.uniform(1000, 5000, 1000),
            'optimization_score': np.random.uniform(0.5, 1.0, 1000)
        })
        
        # Create sample workload data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
        self.workload_data = pd.DataFrame({
            'timestamp': dates,
            'cpu_demand': np.random.uniform(20, 80, 200),
            'memory_demand': np.random.uniform(30, 70, 200),
            'storage_demand': np.random.uniform(40, 90, 200),
            'network_demand': np.random.uniform(25, 75, 200)
        })
    
    def preprocess_data(self):
        # Scale the data for LSTM models
        features = ['cpu_demand', 'memory_demand', 'storage_demand', 'network_demand']
        
        # Create scalers for each feature
        for feature in features:
            self.scalers[feature] = MinMaxScaler(feature_range=(0, 1))
            self.workload_data[f'{feature}_scaled'] = self.scalers[feature].fit_transform(
                self.workload_data[feature].values.reshape(-1, 1)
            )
    
    def create_sequences(self, data, seq_length=12):
        """Create sequences for LSTM model"""
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)].values
            y = data.iloc[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def build_lstm_model(self, feature):
        """Build and train LSTM model for a specific feature"""
        try:
            # Create time series data
            feature_data = self.workload_data[f'{feature}_scaled']
            X, y = self.create_sequences(feature_data)
            
            # Check if we have enough data
            if len(X) < 2 or len(y) < 2:
                self.models[feature] = None
                return None, None, (None, None, None)
            
            # Split data with minimum size check
            train_size = max(1, int(len(X) * 0.8))
            if train_size >= len(X):
                train_size = len(X) - 1
                
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Check if we have valid training data
            if len(X_train) == 0 or len(y_train) == 0:
                self.models[feature] = None
                return None, None, (None, None, None)
            
            # Reshape input to be [samples, time steps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            if len(X_test) > 0:
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Simple model with fewer parameters
            model = Sequential([
                LSTM(units=20, input_shape=(X_train.shape[1], 1)),
                Dense(units=1)
            ])
            epochs = 10
            batch_size = 16
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Train model with validation data only if available
            validation_data = None
            if len(X_test) > 0 and len(y_test) > 0:
                validation_data = (X_test, y_test)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=0
            )
            
            # Predict and evaluate only if we have test data
            predictions = None
            if len(X_test) > 0:
                predictions = model.predict(X_test, verbose=0)
                mse = mean_squared_error(y_test, predictions)
            
            self.models[feature] = model
            return model, history, (X_test, y_test, predictions)
            
        except Exception as e:
            self.models[feature] = None
            return None, None, (None, None, None)
    
    def forecast_resource_usage(self, feature, days=7):
        """Generate forecasts for the next N days"""
        seq_length = 12
        
        # Get last sequence from the data
        last_sequence = self.workload_data[f'{feature}_scaled'].iloc[-seq_length:].values.reshape(1, seq_length, 1)
        
        # Initialize forecasted values list with the last known value
        forecasted_values = [self.workload_data[f'{feature}_scaled'].iloc[-1]]
        
        # Reduce forecast period for faster processing
        hours_to_forecast = 48  # 2 days only
        
        # Make predictions for each hour
        for _ in range(hours_to_forecast):
            # Predict the next value
            next_value = self.models[feature].predict(last_sequence, verbose=0)[0][0]
            forecasted_values.append(next_value)
            
            # Update the sequence by adding the new value and dropping the oldest
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_value
        
        # Inverse transform to get actual values
        forecasted_scaled = np.array(forecasted_values).reshape(-1, 1)
        forecasted_actual = self.scalers[feature].inverse_transform(forecasted_scaled)
        
        # Create a date range for the forecast period
        last_date = self.workload_data['timestamp'].iloc[-1]
        date_range = [last_date + timedelta(hours=i) for i in range(hours_to_forecast + 1)]
        
        return date_range, forecasted_actual
    
    def calculate_resource_allocation(self):
        """Calculate recommended resource allocation based on forecasted usage"""
        resources = {
            'cpu': {},
            'memory': {},
            'storage': {},
            'network': {}
        }
        
        # Map features to resources
        feature_map = {
            'cpu_demand': 'cpu',
            'memory_demand': 'memory',
            'storage_demand': 'storage',
            'network_demand': 'network'
        }
        
        # For each feature, forecast and calculate recommended allocation
        for feature, resource_type in feature_map.items():
            if feature in self.models and self.models[feature] is not None:
                # Get forecasted usage
                dates, forecast = self.forecast_resource_usage(feature)
                
                # Calculate daily averages and peaks
                daily_forecasts = {}
                for i in range(0, len(dates)-1, 24):
                    if i+24 <= len(forecast):
                        day = dates[i].strftime('%Y-%m-%d')
                        daily_values = forecast[i:i+24].flatten()
                        daily_forecasts[day] = {
                            'avg': np.mean(daily_values),
                            'peak': np.max(daily_values),
                            'min': np.min(daily_values)
                        }
                
                resources[resource_type] = {
                    'forecast': forecast.flatten().tolist(),
                    'dates': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
                    'daily': daily_forecasts
                }
            else:
                # Default values if model is not available
                resources[resource_type] = {
                    'forecast': [50] * 48,
                    'dates': [(datetime.now() + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(48)],
                    'daily': {
                        datetime.now().strftime('%Y-%m-%d'): {
                            'avg': 50,
                            'peak': 70,
                            'min': 30
                        }
                    }
                }
        
        return resources
    
    def suggest_resources(self, resource_allocation):
        """Suggest resource types based on predicted allocation"""
        suggestions = {}
        
        # EC2 instance suggestions based on CPU and memory
        cpu_peak = max([day['peak'] for day in resource_allocation['cpu']['daily'].values()])
        memory_peak = max([day['peak'] for day in resource_allocation['memory']['daily'].values()])
        
        # Simple instance mapping based on CPU and memory requirements
        if cpu_peak < 30 and memory_peak < 40:
            suggestions['instance'] = 't3.medium'
        elif cpu_peak < 60 and memory_peak < 60:
            suggestions['instance'] = 'm5.large'
        elif cpu_peak < 90:
            suggestions['instance'] = 'c5.xlarge'
        else:
            suggestions['instance'] = 'c5.2xlarge'
        
        # Storage recommendations
        storage_peak = max([day['peak'] for day in resource_allocation['storage']['daily'].values()])
        if storage_peak < 100:
            suggestions['storage'] = 'gp2 100GB'
        elif storage_peak < 300:
            suggestions['storage'] = 'gp2 300GB'
        else:
            suggestions['storage'] = 'gp2 500GB'
        
        # Network recommendations
        network_peak = max([day['peak'] for day in resource_allocation['network']['daily'].values()])
        if network_peak < 100:
            suggestions['network'] = 'Low - Up to 5 Gbps'
        elif network_peak < 500:
            suggestions['network'] = 'Medium - Up to 10 Gbps'
        else:
            suggestions['network'] = 'High - Up to 25 Gbps'
            
        # Function recommendations
        function_executions = self.resource_data['initial_resource_pool'].mean() * 1000  # Simulated data
        suggestions['function'] = {
            'memory': f"{int(memory_peak * 1.2)}MB",
            'executions_per_day': int(function_executions)
        }
        
        # Database recommendations
        suggestions['database'] = {
            'read_capacity': int(cpu_peak * 0.5),
            'write_capacity': int(cpu_peak * 0.3)
        }
        
        return suggestions
    
    def estimate_costs(self, resource_suggestions):
        """Estimate costs based on resource suggestions"""
        costs = {
            'hourly': {},
            'daily': {},
            'monthly': {}
        }
        
        # Instance pricing (simplified)
        instance_prices = {
            't3.medium': 0.0416,
            'm5.large': 0.096,
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34
        }
        
        # Storage pricing (simplified)
        storage_prices = {
            'gp2 100GB': 0.1 * 100,  # $0.1 per GB
            'gp2 300GB': 0.1 * 300,
            'gp2 500GB': 0.1 * 500
        }
        
        # Function pricing
        function_price_per_million = 0.20  # $ per million executions
        function_price_per_gb_second = 0.0000166667  # $ per GB-second
        
        # Calculate instance costs
        instance_hourly = instance_prices[resource_suggestions['instance']]
        costs['hourly']['instance'] = instance_hourly
        costs['daily']['instance'] = instance_hourly * 24
        costs['monthly']['instance'] = instance_hourly * 24 * 30
        
        # Calculate storage costs (monthly)
        storage_type = resource_suggestions['storage']
        storage_monthly = storage_prices[storage_type] / 30
        costs['hourly']['storage'] = storage_monthly / 24
        costs['daily']['storage'] = storage_monthly / 30
        costs['monthly']['storage'] = storage_monthly
        
        # Calculate function costs
        function_executions = resource_suggestions['function']['executions_per_day']
        function_memory = int(resource_suggestions['function']['memory'].replace('MB', ''))
        function_daily_cost = (function_executions / 1000000) * function_price_per_million
        function_daily_cost += (function_executions * (function_memory / 1024) * 0.1 * function_price_per_gb_second)
        
        costs['hourly']['function'] = function_daily_cost / 24
        costs['daily']['function'] = function_daily_cost
        costs['monthly']['function'] = function_daily_cost * 30
        
        # Calculate database costs
        database_read = resource_suggestions['database']['read_capacity']
        database_write = resource_suggestions['database']['write_capacity']
        database_monthly = (database_read * 0.00013 + database_write * 0.00065) * 24 * 30
        
        costs['hourly']['database'] = database_monthly / (24 * 30)
        costs['daily']['database'] = database_monthly / 30
        costs['monthly']['database'] = database_monthly
        
        # Calculate network costs (simplified)
        network_type = resource_suggestions['network']
        if 'Low' in network_type:
            network_cost = 0.05 * 100  # $0.05 per GB, assuming 100GB transfer
        elif 'Medium' in network_type:
            network_cost = 0.05 * 300
        else:
            network_cost = 0.05 * 500
            
        costs['hourly']['network'] = network_cost / (24 * 30)
        costs['daily']['network'] = network_cost / 30
        costs['monthly']['network'] = network_cost
        
        # Calculate totals
        for period in costs:
            costs[period]['total'] = sum(costs[period].values())
            
        return costs
    
    def generate_autoscaling_recommendations(self, resource_allocation):
        """Generate auto-scaling recommendations based on resource patterns"""
        recommendations = {}
        
        # Analyze CPU patterns
        cpu_daily = resource_allocation['cpu']['daily']
        cpu_values = np.array([day['avg'] for day in cpu_daily.values()])
        cpu_peaks = np.array([day['peak'] for day in cpu_daily.values()])
        cpu_variance = np.std(cpu_values) / np.mean(cpu_values)
        
        # Determine if the workload is steady or variable
        if cpu_variance < 0.2:  # Low variance
            recommendations['scaling_policy'] = "Scheduled Scaling"
            recommendations['explanation'] = "The workload shows predictable patterns with low variance. Use scheduled scaling to adjust capacity at predetermined times."
        else:  # High variance
            recommendations['scaling_policy'] = "Target Tracking"
            recommendations['explanation'] = "The workload shows variable patterns. Use target tracking to adjust capacity based on actual resource utilization."
        
        # Calculate the peak-to-average ratio
        peak_avg_ratio = np.mean(cpu_peaks) / np.mean(cpu_values)
        
        # Set the scaling thresholds
        recommendations['scale_out_threshold'] = int(min(80, max(60, np.mean(cpu_values) + np.std(cpu_values))))
        recommendations['scale_in_threshold'] = int(min(40, max(20, np.mean(cpu_values) - np.std(cpu_values))))
        
        # Determine the cool-down periods
        if peak_avg_ratio > 1.5:  # High peaks
            recommendations['scale_out_cooldown'] = 60  # 1 minute
            recommendations['scale_in_cooldown'] = 300  # 5 minutes
        else:
            recommendations['scale_out_cooldown'] = 180  # 3 minutes
            recommendations['scale_in_cooldown'] = 600  # 10 minutes
        
        # Set min/max capacity
        avg_cpu = np.mean(cpu_values)
        min_capacity = max(1, int(avg_cpu / 50))  # Each instance can handle ~50% CPU
        max_capacity = max(2, int(max(cpu_peaks) / 40))  # Each instance should run at max 80% during peaks
        
        recommendations['min_capacity'] = min_capacity
        recommendations['max_capacity'] = max_capacity
        
        return recommendations
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        self.load_data()
        self.preprocess_data()
        
        for feature in ['cpu_demand', 'memory_demand', 'storage_demand', 'network_demand']:
            self.build_lstm_model(feature)
        
        resource_allocation = self.calculate_resource_allocation()
        resource_suggestions = self.suggest_resources(resource_allocation)
        cost_estimates = self.estimate_costs(resource_suggestions)
        autoscaling_recommendations = self.generate_autoscaling_recommendations(resource_allocation)
        
        return {
            'resource_allocation': resource_allocation,
            'resource_suggestions': resource_suggestions,
            'cost_estimates': cost_estimates,
            'autoscaling_recommendations': autoscaling_recommendations
        }
    
    def get_cost_optimized_suggestions(self, user_input):
        """Get cost-optimized suggestions based on user input"""
        # Load data if not already loaded
        if self.resource_data is None:
            self.load_data()
            self.preprocess_data()
        
        # Extract user input
        cpu_demand = user_input['cpu_demand']
        memory_demand = user_input['memory_demand']
        storage_demand = user_input['storage_demand']
        network_demand = user_input['network_demand']
        application_type = user_input['application_type']
        deployment_region = user_input['deployment_region']
        
        # Create resource allocation based on user input
        resource_allocation = {
            'cpu': {
                'daily': {
                    'day1': {'avg': cpu_demand, 'peak': cpu_demand * 1.2, 'min': cpu_demand * 0.8}
                }
            },
            'memory': {
                'daily': {
                    'day1': {'avg': memory_demand, 'peak': memory_demand * 1.2, 'min': memory_demand * 0.8}
                }
            },
            'storage': {
                'daily': {
                    'day1': {'avg': storage_demand, 'peak': storage_demand * 1.1, 'min': storage_demand * 0.9}
                }
            },
            'network': {
                'daily': {
                    'day1': {'avg': network_demand, 'peak': network_demand * 1.3, 'min': network_demand * 0.7}
                }
            }
        }
        
        # Get resource suggestions
        resource_suggestions = self.suggest_resources(resource_allocation)
        
        # Get cost estimates
        cost_estimates = self.estimate_costs(resource_suggestions)
        
        # Get autoscaling recommendations
        autoscaling_recommendations = self.generate_autoscaling_recommendations(resource_allocation)
        
        # Generate application-specific recommendations
        app_specific_recommendations = self._get_app_specific_recommendations(application_type)
        
        # Generate region-specific recommendations
        region_specific_recommendations = self._get_region_specific_recommendations(deployment_region)
        
        return {
            'resource_suggestions': resource_suggestions,
            'cost_estimates': cost_estimates,
            'autoscaling_recommendations': autoscaling_recommendations,
            'app_specific_recommendations': app_specific_recommendations,
            'region_specific_recommendations': region_specific_recommendations
        }
    
    def _get_app_specific_recommendations(self, application_type):
        """Get application-specific recommendations"""
        recommendations = []
        
        if application_type == "Web Application":
            recommendations = [
                "Consider using Elastic Beanstalk for easier deployment and scaling",
                "Use CDN for content delivery to reduce latency",
                "Consider using object storage for static content storage",
                "Use managed database for database needs with auto-scaling"
            ]
        elif application_type == "Mobile Backend":
            recommendations = [
                "Consider using amplify for backend services",
                "Use NoSQL database for flexible data storage",
                "Implement API gateway for API management",
                "Use authentication services for user authentication"
            ]
        elif application_type == "Machine Learning":
            recommendations = [
                "Consider machine learning platforms for model training and deployment",
                "Use spot instances for cost-effective training",
                "Implement storage for model artifact storage",
                "Use serverless for inference"
            ]
        else:
            recommendations = [
                "Consider using managed services to reduce operational overhead",
                "Implement auto-scaling to match demand",
                "Use monitoring for monitoring and alerting",
                "Consider using configuration management"
            ]
        
        return recommendations

    def _get_region_specific_recommendations(self, deployment_region):
        """Get region-specific recommendations"""
        recommendations = []
        
        if deployment_region in ["us-east-1", "us-west-2"]:
            recommendations = [
                "These regions offer the widest range of services and often the best pricing",
                "Consider using reserved instances for long-term cost savings",
                "Look into savings plans for flexible pricing options"
            ]
        elif deployment_region in ["eu-west-1", "eu-central-1"]:
            recommendations = [
                "Consider multi-region deployment for EU data compliance",
                "Use DNS for global traffic routing",
                "Look into web application firewall for security at the edge"
            ]
        elif deployment_region in ["ap-southeast-1", "ap-south-1"]:
            recommendations = [
                "Consider multi-region deployment for global users",
                "Use DNS for global traffic routing",
                "Implement CDN for content delivery to reduce latency"
            ]
        else:
            recommendations = [
                "Check for region-specific pricing and service availability",
                "Consider using global accelerator for improved performance",
                "Look into transit gateway for multi-region networking"
            ]
        
        return recommendations

    def predict_prices(self, application_type, deployment_region, application_name=""):
        """Predict minimum viable prices based on application name, type and region"""
        
        def analyze_app_name(name, app_type):
            """Analyze application name for keywords to adjust resource needs, considering app type"""
            if not name:
                return 1.0  # Default multiplier if no name provided
                
            name = name.lower()
            multiplier = 1.0
            keywords_found = []
            
            # Size indicators - apply more dramatic differences
            if any(word in name for word in ['small', 'tiny', 'mini', 'basic', 'todo', 'simple']):
                multiplier *= 0.5  # Much smaller for "small" apps
                keywords_found.append(f"small/basic ({0.5}x)")
            elif any(word in name for word in ['large', 'enterprise', 'pro', 'premium', 'advanced']):
                multiplier *= 2.0  # Much larger for "enterprise" apps
                keywords_found.append(f"large/enterprise (2.0x)")
            
            # Application name keywords with stronger multipliers
            keyword_multipliers = {
                # General keywords
                'free': 0.4,
                'lite': 0.6,
                'basic': 0.5,
                'standard': 1.0,
                'premium': 1.8,
                'enterprise': 2.2,
                'professional': 1.7,
                
                # Function keywords
                'analytics': 1.6,
                'dashboard': 1.4,
                'admin': 1.3,
                'monitor': 1.5,
                'report': 1.4,
                'ai': 2.0,
                'ml': 2.0,
                'predict': 1.7,
                'learn': 1.8,
                'vision': 2.2,
                'voice': 1.9,
                'search': 1.6,
                'index': 1.5,
                'chat': 1.4,
                'message': 1.3,
                'social': 1.5,
                'stream': 1.8,
                'video': 2.1,
                'audio': 1.6,
                'media': 1.7,
                'game': 2.0,
                'shop': 1.7,
                'store': 1.7,
                'ecommerce': 1.8,
                'payment': 1.6,
                'finance': 1.7,
                'bank': 1.9,
                'secure': 1.5,
                'auth': 1.3,
                'user': 1.2,
                'content': 1.1,
                'blog': 0.8,
                'cms': 0.9,
                'static': 0.7,
                'data': 1.5,
                'database': 1.4,
                'storage': 1.3,
                'file': 1.2,
                'api': 1.4,
                'gateway': 1.5,
                'service': 1.3,
                'micro': 1.1,
                'web': 1.0,
                'mobile': 1.2,
                'app': 1.1,
                'cloud': 1.3,
                'serverless': 1.2,
                'container': 1.4,
                'docker': 1.4,
                'kubernetes': 1.6,
                'devops': 1.5,
                'ci': 1.4,
                'cd': 1.4,
                'test': 0.9,
                'dev': 0.8,
                'prod': 1.5,
                'staging': 1.1,
                'beta': 0.9
            }
            
            # Check for each keyword in the app name
            applied_keywords = []
            for keyword, keyword_multiplier in keyword_multipliers.items():
                if keyword in name:
                    multiplier *= keyword_multiplier
                    applied_keywords.append(f"{keyword} ({keyword_multiplier}x)")
                    # Apply stronger effect for keywords appearing at the start of name
                    if name.startswith(keyword):
                        multiplier *= 1.2
                        applied_keywords.append(f"{keyword} at start (1.2x)")
            
            # Apply stronger type-specific multipliers
            type_specific_multipliers = {
                'Web Application': {
                    'ecommerce': 2.0,
                    'shop': 1.8,
                    'store': 1.8,
                    'cms': 0.7,
                    'blog': 0.6,
                    'content': 0.7,
                    'portfolio': 0.8,
                    'corporate': 1.5,
                    'booking': 1.6,
                    'reservation': 1.7
                },
                'Mobile Backend': {
                    'game': 2.2,
                    'ar': 2.5,
                    'vr': 2.5,
                    'social': 1.8,
                    'chat': 1.7,
                    'messaging': 1.7,
                    'location': 1.5,
                    'map': 1.6,
                    'tracker': 1.4,
                    'fitness': 1.3
                },
                'Machine Learning': {
                    'training': 2.5,
                    'inference': 1.6,
                    'vision': 2.3,
                    'image': 2.0,
                    'video': 2.4,
                    'nlp': 2.0,
                    'language': 1.9,
                    'speech': 2.1,
                    'recommend': 1.7
                }
            }
            
            # Apply type-specific multipliers
            if app_type in type_specific_multipliers:
                for keyword, type_multiplier in type_specific_multipliers[app_type].items():
                    if keyword in name:
                        multiplier *= type_multiplier
                        applied_keywords.append(f"{keyword} ({type_multiplier}x for {app_type})")
            
            # Apply scale-based multiplier based on name length (longer names often mean more complex apps)
            name_length_factor = min(max(len(name) / 20, 0.8), 1.5)  # Between 0.8 and 1.5
            multiplier *= name_length_factor
            
            # Apply more noticeable region-specific adjustments to the name multiplier
            region_word_multipliers = {
                'global': 1.5,
                'enterprise': 1.8,
                'corporate': 1.6,
                'startup': 0.8,
                'personal': 0.6,
                'hobby': 0.5,
                'school': 0.6,
                'university': 0.9,
                'government': 1.7,
                'healthcare': 1.8,
                'finance': 1.9,
                'bank': 2.0,
                'retail': 1.5,
                'education': 1.0,
                'nonprofit': 0.8,
                'media': 1.5,
                'entertainment': 1.7,
                'travel': 1.6,
                'hospitality': 1.5,
                'restaurant': 1.2,
                'food': 1.3
            }
            
            for sector, sector_multiplier in region_word_multipliers.items():
                if sector in name:
                    multiplier *= sector_multiplier
                    applied_keywords.append(f"{sector} sector ({sector_multiplier}x)")
            
            # Ensure the multiplier has a reasonable range
            multiplier = min(max(multiplier, 0.4), 5.0)
            
            return multiplier
        
        # Base costs for different instance types (monthly)
        instance_costs = {
            't3.nano': 4.25,
            't3.micro': 8.50,
            't3.small': 17.00,
            't3.medium': 34.00,
            't3.large': 68.00,
            'c5.large': 85.00,
            'r5.large': 126.00,
            'p3.2xlarge': 918.00
        }
        
        # Base configurations for each application type
        app_configs = {
            'Web Application': {
                'tiers': {
                    'basic': {
                        'instance_type': 't3.nano',
                        'cost_multiplier': 0.8,
                        'storage_gb': 10,
                        'function_calls': 5000,
                        'network_gb': 20
                    },
                    'standard': {
                        'instance_type': 't3.micro',
                        'cost_multiplier': 1.0,
                        'storage_gb': 20,
                        'function_calls': 10000,
                        'network_gb': 50
                    },
                    'premium': {
                        'instance_type': 't3.small',
                        'cost_multiplier': 1.2,
                        'storage_gb': 50,
                        'function_calls': 25000,
                        'network_gb': 100
                    }
                }
            },
            'Mobile Backend': {
                'tiers': {
                    'basic': {
                        'instance_type': 't3.micro',
                        'cost_multiplier': 1.0,
                        'storage_gb': 20,
                        'function_calls': 15000,
                        'network_gb': 50
                    },
                    'standard': {
                        'instance_type': 't3.small',
                        'cost_multiplier': 1.2,
                        'storage_gb': 50,
                        'function_calls': 30000,
                        'network_gb': 100
                    },
                    'premium': {
                        'instance_type': 't3.medium',
                        'cost_multiplier': 1.5,
                        'storage_gb': 100,
                        'function_calls': 60000,
                        'network_gb': 200
                    }
                }
            },
            'Machine Learning': {
                'tiers': {
                    'basic': {
                        'instance_type': 't3.large',
                        'cost_multiplier': 1.5,
                        'storage_gb': 100,
                        'function_calls': 10000,
                        'network_gb': 100
                    },
                    'standard': {
                        'instance_type': 'p3.2xlarge',
                        'cost_multiplier': 2.0,
                        'storage_gb': 500,
                        'function_calls': 25000,
                        'network_gb': 200
                    },
                    'premium': {
                        'instance_type': 'p3.2xlarge',
                        'cost_multiplier': 3.0,
                        'storage_gb': 1000,
                        'function_calls': 50000,
                        'network_gb': 500
                    }
                }
            }
        }
        
        # Region-specific cost multipliers
        region_multipliers = {
            'us-east-1': 1.0,
            'us-west-2': 1.05,
            'eu-west-1': 1.12,
            'ap-southeast-1': 1.15,
            'ap-south-1': 0.92
        }
        
        # Get base configuration for the selected application type
        app_type_config = app_configs.get(application_type, app_configs['Web Application'])
        
        # Determine tier based on application name analysis
        name_multiplier = analyze_app_name(application_name, application_type)
        
        # Select appropriate tier based on the name analysis with more dramatic thresholds
        if name_multiplier <= 0.75:
            tier = 'basic'
        elif name_multiplier >= 1.5:
            tier = 'premium'
        else:
            tier = 'standard'
            
        tier_config = app_type_config['tiers'][tier]
        
        # Apply region-specific multiplier
        region_multiplier = region_multipliers.get(deployment_region, 1.0)
        
        # Calculate base costs with tier-specific configuration
        instance_type = tier_config['instance_type']
        base_instance_cost = instance_costs[instance_type]
        
        # Calculate service-specific costs with more variation
        instance_cost = base_instance_cost * tier_config['cost_multiplier'] * region_multiplier * name_multiplier
        database_cost = (base_instance_cost * 0.35) * tier_config['cost_multiplier'] * region_multiplier * name_multiplier
        storage_cost = (tier_config['storage_gb'] * 0.023) * region_multiplier * name_multiplier
        function_cost = (tier_config['function_calls'] / 1000000) * 0.20 * region_multiplier * name_multiplier
        network_cost = (tier_config['network_gb'] * 0.09) * region_multiplier * name_multiplier
        
        # Extra service costs based on application type and keywords
        extra_services = {}
        
        # Add appropriate extra services based on app type
        if application_type == 'Web Application':
            # Add CDN for content delivery
            cdn_cost = (tier_config['network_gb'] * 0.085) * name_multiplier
            extra_services['cdn'] = round(cdn_cost, 2)
            
            # Add more for ecommerce sites
            if any(word in application_name.lower() for word in ['ecommerce', 'shop', 'store']):
                db_catalog_cost = 20 * name_multiplier  # For product catalog
                extra_services['db_catalog'] = round(db_catalog_cost, 2)
        
        elif application_type == 'Mobile Backend':
            # Add notification service
            notification_cost = (tier_config['function_calls'] / 10000) * 0.005 * name_multiplier
            extra_services['notifications'] = round(notification_cost, 2)
            
            # Add authentication for mobile apps
            auth_cost = 15 * name_multiplier
            extra_services['auth'] = round(auth_cost, 2)
        
        elif application_type == 'Machine Learning':
            # Add ML specific services
            ml_platform_cost = 120 * name_multiplier
            extra_services['ml_platform'] = round(ml_platform_cost, 2)
            
            # Add more for vision or NLP
            if any(word in application_name.lower() for word in ['vision', 'image', 'video', 'nlp', 'language']):
                vision_cost = 45 * name_multiplier
                extra_services['vision_api'] = round(vision_cost, 2)
        
        # Format predictions
        predictions = {
            'ec2': {
                'current': round(instance_cost, 2),
                'previous': round(instance_cost * 0.9, 2),
                'change': round(instance_cost * 0.1, 2)
            },
            'rds': {
                'current': round(database_cost, 2),
                'previous': round(database_cost * 0.9, 2),
                'change': round(database_cost * 0.1, 2)
            },
            's3': {
                'current': round(storage_cost, 2),
                'previous': round(storage_cost * 0.9, 2),
                'change': round(storage_cost * 0.1, 2)
            },
            'lambda': {
                'current': round(function_cost, 2),
                'previous': round(function_cost * 0.9, 2),
                'change': round(function_cost * 0.1, 2)
            },
            'network': {
                'current': round(network_cost, 2),
                'previous': round(network_cost * 0.9, 2),
                'change': round(network_cost * 0.1, 2)
            }
        }
        
        # Add extra services to predictions
        for service, cost in extra_services.items():
            predictions[service] = {
                'current': cost,
                'previous': round(cost * 0.9, 2),
                'change': round(cost * 0.1, 2)
            }
        
        return predictions

# Initialize Streamlit app
def main():
    st.set_page_config(
        page_title="Cloud Resource Analysis",
        page_icon="☁️",
        layout="centered"
    )
    
    st.title("Cloud Resource Analysis")
    st.write("Calculate optimal resources based on your application profile")
    
    # Initialize the analyzer
    analyzer = ResourceAnalyzer()
    
    # Run the analysis in background (or cache)
    if 'analysis_results' not in st.session_state:
        with st.spinner("Initializing..."):
            try:
                st.session_state.analysis_results = analyzer.run_analysis()
            except Exception as e:
                st.error(f"Error running analysis: {e}")
                return
    
    # Simple input form
    with st.form("application_form"):
        st.markdown("### Application Details")
        
        application_name = st.text_input(
            "Application Name",
            help="Enter your application name (e.g., 'small-blog', 'enterprise-ecommerce', 'ml-pipeline')"
        )
        
        application_type = st.selectbox(
            "Application Type",
            ["Web Application", "Mobile Backend", "Machine Learning"],
            help="Select your application type"
        )
        
        deployment_region = st.selectbox(
            "Deployment Region",
            ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-south-1"],
            help="Select your deployment region"
        )
        
        submitted = st.form_submit_button("Calculate Resources", type="primary")
        
        if submitted:
            if not application_name:
                st.error("Please enter an application name")
                return
                
            st.session_state.form_data = {
                'application_name': application_name,
                'application_type': application_type,
                'deployment_region': deployment_region
            }
    
    # Show results if form is submitted
    if 'form_data' in st.session_state:
        st.markdown("---")
        
        # Get price predictions with application name
        app_name = st.session_state.form_data['application_name']
        price_predictions = analyzer.predict_prices(
            st.session_state.form_data['application_type'],
            st.session_state.form_data['deployment_region'],
            app_name
        )
        
        # Determine tier based on the name analysis
        name = app_name.lower()
        
        # Size indicators
        if any(word in name for word in ['small', 'tiny', 'mini', 'basic', 'todo', 'todos']):
            tier = "Basic Tier"
        elif any(word in name for word in ['large', 'enterprise', 'pro', 'premium']):
            tier = "Premium Tier"
        # Functionality indicators that suggest premium resources
        elif any(keyword in name for keyword in ['ecommerce', 'e-commerce', 'shop', 'store', 'commerce', 'ai', 'ml', 'analytics']):
            tier = "Premium Tier"
        else:
            tier = "Standard Tier"
        
        # Display the cost analysis header with application details
        st.markdown(f"### Cost Analysis for {st.session_state.form_data['application_name']}")
        st.markdown(f"**Application Type:** {st.session_state.form_data['application_type']} | **Region:** {st.session_state.form_data['deployment_region']}")
        st.markdown(f"**Service Tier:** {tier}")
        
        # Calculate monthly costs for each service
        ec2_cost = price_predictions['ec2']['current']
        rds_cost = price_predictions['rds']['current']
        s3_cost = price_predictions['s3']['current']
        lambda_cost = price_predictions['lambda']['current']
        network_cost = price_predictions['network']['current']
        
        # Calculate total monthly cost
        total_cost = sum(price_predictions[service]['current'] for service in price_predictions)
        
        # Display Monthly Costs in card layout similar to cost explorer
        st.markdown("### Monthly Cost Breakdown")
        
        # Custom CSS for the cards
        st.markdown("""
        <style>
        .cost-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .cost-card h4 {
            margin: 0;
            color: #444;
        }
        .cost-card p {
            margin: 5px 0;
            font-size: 24px;
            font-weight: bold;
            color: #0077b6;
        }
        .cost-card small {
            color: #666;
        }
        .total-cost {
            background-color: #e6f3ff;
            border: 2px solid #0077b6;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        .total-cost h3 {
            margin: 0;
            color: #444;
        }
        .total-cost p {
            margin: 10px 0;
            font-size: 32px;
            font-weight: bold;
            color: #00a86b;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create two rows of service cost cards
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        row3_col1, row3_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown(f"""
            <div class="cost-card">
                <h4>🖥️ Instance</h4>
                <p>${ec2_cost:.2f}/month</p>
                <small>1 resource</small>
            </div>
            """, unsafe_allow_html=True)
            
        with row1_col2:
            st.markdown(f"""
            <div class="cost-card">
                <h4>🗄️ Database</h4>
                <p>${rds_cost:.2f}/month</p>
                <small>1 resource</small>
            </div>
            """, unsafe_allow_html=True)
            
        with row2_col1:
            st.markdown(f"""
            <div class="cost-card">
                <h4>📦 Storage</h4>
                <p>${s3_cost:.2f}/month</p>
                <small>1 resource</small>
            </div>
            """, unsafe_allow_html=True)
            
        with row2_col2:
            st.markdown(f"""
            <div class="cost-card">
                <h4>⚡ Functions</h4>
                <p>${lambda_cost:.2f}/month</p>
                <small>1 resource</small>
            </div>
            """, unsafe_allow_html=True)
            
        with row3_col1:
            st.markdown(f"""
            <div class="cost-card">
                <h4>🌐 Network</h4>
                <p>${network_cost:.2f}/month</p>
                <small>1 resource</small>
            </div>
            """, unsafe_allow_html=True)
            
        with row3_col2:
            st.markdown(f"""
            <div class="cost-card">
                <h4>🔄 Auto Scaling</h4>
                <p>$0.00/month</p>
                <small>Included</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Display total cost in a framed box
        st.markdown(f"""
        <div class="total-cost">
            <h3>Total Monthly Cost</h3>
            <p>${total_cost:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 