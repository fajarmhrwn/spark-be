from flask import Blueprint, jsonify, request
from client import Dataiku
from dotenv import load_dotenv
import pandas as pd
import random
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import traceback
from datetime import datetime, timedelta
import dataikuscoring.load as loader

load_dotenv()

fetch_api = Blueprint("fetch_api", __name__, url_prefix="/api")
client = Dataiku().client


@fetch_api.route("/hello", methods=["GET"])
def hello():
    return jsonify({"key": "hello"})

@fetch_api.route("/datasets", methods=["GET"])
def datasets():
    try:
        print(client.get_auth_info())
        project = client.get_default_project()
        datasets = project.list_datasets()

        # Extract name and type for each dataset
        dataset_list = []
        for d in datasets:
            dataset_list.append({
                "name": d.name
            })

        return jsonify(dataset_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@fetch_api.route("/production-stat-area", methods=["GET"])
def productionStatArea():
    try:
        area = request.args.get('area', default=None, type=str)
        # 1️⃣ Fetch dataset
        project = client.get_default_project()
        dataset = project.get_dataset("Well_data")
        df = dataset.get_as_core_dataset()
        df = df.get_dataframe()

        '''
          {
            current : {
                gas: 100,
                oil: 200,
                water: 300
            },
            previous : {
                gas: 90,
                oil: 180,
                water: 270
            },
          }
        '''
        # DATE column is in 2025-05-26 00:00:00 format, we need to convert it to YYYY-MM-DD format
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
        # Filter out rows where DATE is NaN
        df = df.dropna(subset=['DATE'])
        # add column WELL_AREA
        df['WELL_AREA'] = df['WELL'].str.split('-').str[0]
        filtered_df = df
        if area and area != "all" and area != "All":
            filtered_df = df[df['WELL_AREA'] == area]
        metric = {}
        today = pd.to_datetime("today").normalize()
        current_date = today.strftime("%Y-%m-%d")
        previous_date = (today - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        current_data = filtered_df[filtered_df['DATE'] == current_date]
        previous_data = filtered_df[filtered_df['DATE'] == previous_date]
        metric['current'] = {
            "gas": current_data['GAS_RATE (MMscf/d)'].sum(),
            "water": current_data['WATER_RATE (stb/d)'].sum(),
            "oil": current_data['OIL_RATE (stb/d)'].sum()
        }
        metric['previous'] = {
            "gas": previous_data['GAS_RATE (MMscf/d)'].sum(),
            "water": previous_data['WATER_RATE (stb/d)'].sum(),
            "oil": previous_data['OIL_RATE (stb/d)'].sum()
        }


        '''     get gas,water production for each date from 30 days before today      '''
        line_chart = {}
        for i in range(30):
            date = (today - pd.DateOffset(days=i)).strftime("%Y-%m-%d")
            daily_data = filtered_df[filtered_df['DATE'] == date]
            if not daily_data.empty:
              line_chart[str(date)] = {
                  "gas": daily_data['GAS_RATE (MMscf/d)'].sum(),
                  "water": daily_data['WATER_RATE (stb/d)'].sum(),
                  "oil": daily_data['OIL_RATE (stb/d)'].sum()
              }

        '''  pie chart data for gas, water, oil production  get the sum of each well with data is today'''
        pie_chart = {}
        for area in df['WELL_AREA'].unique():
            area_data = df[(df['WELL_AREA'] == area) & (df['DATE'] == current_date)]
            if not area_data.empty:
                pie_chart[area] = {
                    "gas": area_data['GAS_RATE (MMscf/d)'].sum(),
                    "water": area_data['WATER_RATE (stb/d)'].sum(),
                    "oil": area_data['OIL_RATE (stb/d)'].sum()
                }
            else:
              pie_chart[area] = {
                  "gas": 0,
                  "water": 0,
                  "oil": 0
              }

        return jsonify({
                "metric": metric,
                "lineChart": line_chart,
                "pieChart": pie_chart
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@fetch_api.route("/production-stat", methods=["GET"])
def productionStat():
    try:
        well = request.args.get('well', default=None, type=str)
        # 1️⃣ Fetch dataset
        project = client.get_default_project()
        dataset = project.get_dataset("Well_data")
        df = dataset.get_as_core_dataset()
        df = df.get_dataframe()

        '''
          {
            current : {
                gas: 100,
                oil: 200,
                water: 300
            },
            previous : {
                gas: 90,
                oil: 180,
                water: 270
            },
          }
        '''
        # DATE column is in 2025-05-26 00:00:00 format, we need to convert it to YYYY-MM-DD format
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
        # Filter out rows where DATE is NaN
        df = df.dropna(subset=['DATE'])
        filtered_df = df
        if well and well != "all" and well != "All":
            filtered_df = df[df['WELL'] == well]
        metric = {}
        today = pd.to_datetime("today").normalize()
        current_date = today.strftime("%Y-%m-%d")
        previous_date = (today - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        current_data = filtered_df[filtered_df['DATE'] == current_date]
        previous_data = filtered_df[filtered_df['DATE'] == previous_date]
        metric['current'] = {
            "gas": current_data['GAS_RATE (MMscf/d)'].sum(),
            "water": current_data['WATER_RATE (stb/d)'].sum(),
            "oil": current_data['OIL_RATE (stb/d)'].sum(),
            "pressure": current_data['DOWNHOLE_PRESSURE (psi)'].mean(),
        }
        metric['previous'] = {
            "gas": previous_data['GAS_RATE (MMscf/d)'].sum(),
            "water": previous_data['WATER_RATE (stb/d)'].sum(),
            "oil": previous_data['OIL_RATE (stb/d)'].sum(),
            "pressure": previous_data['DOWNHOLE_PRESSURE (psi)'].mean(),
        }


        '''     get gas,water production for each date from 30 days before today      '''
        line_chart = {}
        for i in range(30):
            date = (today - pd.DateOffset(days=i)).strftime("%Y-%m-%d")
            daily_data = filtered_df[filtered_df['DATE'] == date]
            if not daily_data.empty:
              line_chart[str(date)] = {
                  "gas": daily_data['GAS_RATE (MMscf/d)'].sum(),
                  "water": daily_data['WATER_RATE (stb/d)'].sum(),
                  "oil": daily_data['OIL_RATE (stb/d)'].sum(),
                  "pressure": daily_data['DOWNHOLE_PRESSURE (psi)'].mean(),
              }

        '''  pie chart data for gas, water, oil production  get the sum of each well with data is today'''
        pie_chart = {}
        for well in df['WELL'].unique():
            well_data = df[(df['WELL'] == well) & (df['DATE'] == current_date)]
            if not well_data.empty:
                pie_chart[well] = {
                    "gas": well_data['GAS_RATE (MMscf/d)'].sum(),
                    "water": well_data['WATER_RATE (stb/d)'].sum(),
                    "oil": well_data['OIL_RATE (stb/d)'].sum()
                }
            else:
              pie_chart[well] = {
                  "gas": 0,
                  "water": 0,
                  "oil": 0
              }

        return jsonify({
                "metric": metric,
                "lineChart": line_chart,
                "pieChart": pie_chart
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@fetch_api.route("/well", methods=["GET"])
def get_well():
  try:
    project = client.get_default_project()
    dataset = project.get_dataset("scored_MHI_distinct")
    df = dataset.get_as_core_dataset()
    df = df.get_dataframe()
    dataset2 = project.get_dataset("Well_data")
    df2 = dataset2.get_as_core_dataset()
    df2 = df2.get_dataframe()
    df2['DATE'] = pd.to_datetime(df2['DATE']).dt.strftime('%Y-%m-%d')
    wells = df2['WELL'].unique()
    today = pd.to_datetime("today").normalize()
    current_date = today.strftime("%Y-%m-%d")
    response = {}
    for name in wells:
      area = name.split('-')[0]
      type_well = "Oil"
      current_prod_well = df2[(df2['WELL'] == name) & (df2['DATE'] == current_date)]
      if not current_prod_well.empty and (current_prod_well['OIL_RATE (stb/d)'].iloc[0] == 0 or pd.isna(current_prod_well['OIL_RATE (stb/d)'].iloc[0])):
        type_well = "Gas"
      # Artificial Lift
      artificial_lift = None
      artificial_status = random.randint(1,3)
      if artificial_status == 1:
         artificial_lift = "Good"
      elif artificial_status == 2:
         artificial_lift = "Warning"
      else:
        artificial_lift = "Critical"

      # Randomize Well Integrity Status 1 - 3
      integrity_status = random.randint(1, 3)
      if integrity_status == 1:
         integrity_status = "Good"
      elif integrity_status == 2:
         integrity_status = "Warning"
      else:
         integrity_status = "Critical"

      workover_score = df[df["WELL"]==name]["workover_score"]
      if workover_score.empty:
         # randomize workover score between 0 and 1
         workover_score = random.uniform(0, 1)
      else:
         workover_score = workover_score.iloc[0]

      response[name] = {
        "area": area,
        "type": type_well,
        "artificialLift": artificial_lift,
        "integrityStatus": integrity_status,
        "workoverScore": workover_score
      }

    return jsonify(response)

  except Exception as e:
    return jsonify({"error": str(e)}), 500

@fetch_api.route("/forecast-oil", methods=["POST"])
def forecast_well():
    try:
        data = request.get_json()
        well = data.get('dataset')
        well_id = data.get('well_id')
        model_type = data.get('model_type')
        forecast_days = int(data.get('forecast_days'))

        project = client.get_default_project()
        dataset = project.get_dataset(well)
        df = dataset.get_as_core_dataset()
        df = df.get_dataframe()

        PROJECT_KEY = "SPARK"
        # Validate required columns
        required_columns = ['DATE', 'OIL_RATE (stb/d)']
        if well_id:
            required_columns.append('WELL')

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                'success': False,
                'error': f'Missing required columns: {missing_columns}',
                'available_columns': list(df.columns)
            }

        # Filter by well if specified
        if well_id:
            df_for_well_check = df # Use current df for well check before it's filtered
            df = df[df['WELL'] == well_id].copy()

            if df.empty:
                available_wells = []
                if 'WELL' in df_for_well_check.columns:
                    available_wells = list(df_for_well_check['WELL'].unique())
                return {
                    'success': False,
                    'error': f'Well ID "{well_id}" not found in dataset "{dataset}" or yields no data.',
                    'available_wells': available_wells
                }

        if df.empty : # if dataset is empty (or became empty after filtering for a well)
             return {
                'success': False,
                'error': f'No data available for the specified criteria (Dataset: "{dataset}", Well ID: "{well_id}").',
            }


        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE').reset_index(drop=True)

        initial_time = df['DATE'].min()
        df['rel_time'] = (df['DATE'] - initial_time).dt.days

        df_clean = df[df['OIL_RATE (stb/d)'] > 0].copy()

        if df_clean.empty:
            return {
                'success': False,
                'error': 'No valid positive oil rate data found for analysis.',
                'original_rows_in_scope': len(df)
            }

        train_df = df_clean.copy()

        # Decline curve functions
        def hyperbolic_decline(t, qi, Di, b):
            return qi / ((1 + b * Di * t) ** (1/b))

        def exponential_decline(t, qi, Di):
            return qi * np.exp(-Di * t)

        def harmonic_decline(t, qi, Di):
            return qi / (1 + Di * t)

        models_results_internal = {} # Store full model info including function objects
        fitted_models_r2 = []

        # Fit Hyperbolic Decline
        try:
            if len(train_df) < 3: raise ValueError("Not enough data points for hyperbolic fitting (need at least 3).")

            # Ensure initial qi is positive and within reasonable bounds of observed data
            initial_qi_guess = max(train_df["OIL_RATE (stb/d)"].iloc[0], 1e-6) # Avoid zero or negative qi
            max_obs_rate = train_df["OIL_RATE (stb/d)"].max()

            hyp_initial_guess = [initial_qi_guess, 0.001, 0.5]
            hyp_lower_bounds = [1e-9, 1e-9, 1e-9] # qi, Di, b > 0
            hyp_upper_bounds = [max_obs_rate * 5, 1.0, 0.999] # b < 1 for standard hyperbolic

            hyp_params, _ = curve_fit(hyperbolic_decline, train_df['rel_time'].values, train_df["OIL_RATE (stb/d)"].values,
                                      p0=hyp_initial_guess, bounds=(hyp_lower_bounds, hyp_upper_bounds), maxfev=20000)
            qi_hyp, Di_hyp, b_hyp = hyp_params
            hyp_pred = hyperbolic_decline(train_df['rel_time'].values, qi_hyp, Di_hyp, b_hyp)

            models_results_internal['hyperbolic'] = {
                'fitted': True, 'params': {'qi': float(qi_hyp), 'Di': float(Di_hyp), 'b': float(b_hyp)},
                'metrics': {'r2': float(r2_score(train_df["OIL_RATE (stb/d)"], hyp_pred)),
                            'mse': float(mean_squared_error(train_df["OIL_RATE (stb/d)"], hyp_pred)),
                            'mae': float(mean_absolute_error(train_df["OIL_RATE (stb/d)"], hyp_pred))},
                'function': hyperbolic_decline
            }
            if np.isfinite(models_results_internal['hyperbolic']['metrics']['r2']):
                fitted_models_r2.append(('hyperbolic', models_results_internal['hyperbolic']['metrics']['r2']))
        except Exception as e:
            models_results_internal['hyperbolic'] = {'fitted': False, 'error': str(e)}

        # Fit Exponential Decline
        try:
            if len(train_df) < 2: raise ValueError("Not enough data points for exponential fitting (need at least 2).")
            initial_qi_guess = max(train_df["OIL_RATE (stb/d)"].iloc[0], 1e-6)
            max_obs_rate = train_df["OIL_RATE (stb/d)"].max()

            exp_initial_guess = [initial_qi_guess, 0.001]
            exp_lower_bounds = [1e-9, 1e-9] # qi, Di > 0
            exp_upper_bounds = [max_obs_rate * 5, 1.0]

            exp_params, _ = curve_fit(exponential_decline, train_df['rel_time'].values, train_df["OIL_RATE (stb/d)"].values,
                                      p0=exp_initial_guess, bounds=(exp_lower_bounds, exp_upper_bounds), maxfev=10000)
            qi_exp, Di_exp = exp_params
            exp_pred = exponential_decline(train_df['rel_time'].values, qi_exp, Di_exp)

            models_results_internal['exponential'] = {
                'fitted': True, 'params': {'qi': float(qi_exp), 'Di': float(Di_exp)},
                'metrics': {'r2': float(r2_score(train_df["OIL_RATE (stb/d)"], exp_pred)),
                            'mse': float(mean_squared_error(train_df["OIL_RATE (stb/d)"], exp_pred)),
                            'mae': float(mean_absolute_error(train_df["OIL_RATE (stb/d)"], exp_pred))},
                'function': exponential_decline
            }
            if np.isfinite(models_results_internal['exponential']['metrics']['r2']):
                fitted_models_r2.append(('exponential', models_results_internal['exponential']['metrics']['r2']))
        except Exception as e:
            models_results_internal['exponential'] = {'fitted': False, 'error': str(e)}

        # Fit Harmonic Decline
        try:
            if len(train_df) < 2: raise ValueError("Not enough data points for harmonic fitting (need at least 2).")
            initial_qi_guess = max(train_df["OIL_RATE (stb/d)"].iloc[0], 1e-6)
            max_obs_rate = train_df["OIL_RATE (stb/d)"].max()

            har_initial_guess = [initial_qi_guess, 0.001]
            har_lower_bounds = [1e-9, 1e-9] # qi, Di > 0
            har_upper_bounds = [max_obs_rate * 5, 1.0]

            har_params, _ = curve_fit(harmonic_decline, train_df['rel_time'].values, train_df["OIL_RATE (stb/d)"].values,
                                      p0=har_initial_guess, bounds=(har_lower_bounds, har_upper_bounds), maxfev=10000)
            qi_har, Di_har = har_params
            har_pred = harmonic_decline(train_df['rel_time'].values, qi_har, Di_har)

            models_results_internal['harmonic'] = {
                'fitted': True, 'params': {'qi': float(qi_har), 'Di': float(Di_har)},
                'metrics': {'r2': float(r2_score(train_df["OIL_RATE (stb/d)"], har_pred)),
                            'mse': float(mean_squared_error(train_df["OIL_RATE (stb/d)"], har_pred)),
                            'mae': float(mean_absolute_error(train_df["OIL_RATE (stb/d)"], har_pred))},
                'function': harmonic_decline
            }
            if np.isfinite(models_results_internal['harmonic']['metrics']['r2']):
                fitted_models_r2.append(('harmonic', models_results_internal['harmonic']['metrics']['r2']))
        except Exception as e:
            models_results_internal['harmonic'] = {'fitted': False, 'error': str(e)}

        valid_fitted_models = [m for m in fitted_models_r2 if models_results_internal[m[0]]['fitted'] and np.isfinite(m[1])]
        if not valid_fitted_models:
            # Prepare a summary without function objects for the error response
            error_models_summary = {name: {k:v for k,v in info.items() if k != 'function'} for name, info in models_results_internal.items()}
            return {
                'success': False,
                'error': 'No models could be successfully fitted with a valid R2 score.',
                'model_fitting_summary': error_models_summary,
                'available_data_points_for_fit': len(train_df)
            }

        best_model_name = max(valid_fitted_models, key=lambda x: x[1])[0]

        selected_model_name = model_type
        if model_type == 'best' or \
           model_type not in models_results_internal or \
           not models_results_internal[model_type].get('fitted', False) or \
           not np.isfinite(models_results_internal[model_type].get('metrics', {}).get('r2', np.nan)):
            selected_model_name = best_model_name

        selected_model_info = models_results_internal[selected_model_name]

        max_hist_rel_time = train_df['rel_time'].max()
        forecast_time_points = np.linspace(0, max_hist_rel_time + forecast_days, num=200)

        historical_data_for_chart = [{
            'x': row['DATE'].isoformat(), 'y': float(row['OIL_RATE (stb/d)']),
            'type': 'historical', 'rel_time': float(row['rel_time'])
        } for _, row in train_df.iterrows()]

        forecast_points_for_chart = []
        model_func_to_call = selected_model_info['function']
        model_params = selected_model_info['params']

        for t_rel in forecast_time_points:
            date_forecast = (initial_time + pd.Timedelta(days=t_rel)).isoformat()
            rate_forecasted = 0.0
            if selected_model_name == 'hyperbolic':
                rate_forecasted = model_func_to_call(t_rel, **model_params)
            else: # Exponential or Harmonic
                rate_forecasted = model_func_to_call(t_rel, model_params['qi'], model_params['Di'])

            forecast_points_for_chart.append({
                'x': date_forecast, 'y': float(max(0, rate_forecasted)),
                'type': 'historical_fit' if t_rel <= max_hist_rel_time else 'forecast',
                'rel_time': float(t_rel)
            })

        # EUR Calculation (30 years from start of production)
        eur_horizon_days = 30 * 365
        eur_30_years = np.nan
        try:
            qi, Di = model_params['qi'], model_params.get('Di', 0) # Di might not exist if params are malformed
            if Di <= 0 : Di = 1e-9 # Avoid division by zero / log of zero later, use a tiny Di

            if selected_model_name == 'hyperbolic':
                b = model_params['b']
                if abs(b) < 1e-6: # Exponential case
                    eur_30_years = (qi / Di) * (1 - np.exp(-Di * eur_horizon_days))
                elif abs(b - 1.0) < 1e-6: # Harmonic case
                    eur_30_years = (qi / Di) * np.log(1 + Di * eur_horizon_days)
                else: # General hyperbolic
                     eur_30_years = (qi / (Di * (1 - b))) * (1 - (1 + b * Di * eur_horizon_days)**((b-1)/b))
            elif selected_model_name == 'exponential':
                eur_30_years = (qi / Di) * (1 - np.exp(-Di * eur_horizon_days))
            elif selected_model_name == 'harmonic':
                eur_30_years = (qi / Di) * np.log(1 + Di * eur_horizon_days)
        except (OverflowError, ZeroDivisionError, ValueError, KeyError):
            eur_30_years = np.nan


        final_models_summary = {name: {k:v for k,v in info.items() if k != 'function'} for name, info in models_results_internal.items()}

        response = {
            'success': True,
            'well_id_analyzed': well_id if well_id else "All Wells (Aggregated)",
            'dataset_used': f"{PROJECT_KEY}.{dataset}",
            'data_for_chart': {'datasets': [
                {'label': 'Historical Production (Cleaned)', 'data': historical_data_for_chart,
                 'borderColor': 'rgb(75, 192, 192)', 'backgroundColor': 'rgba(75, 192, 192, 0.5)',
                 'pointStyle': 'circle', 'pointRadius': 4, 'showLine': False},
                {'label': f'{selected_model_name.title()} Model Fit & Forecast', 'data': forecast_points_for_chart,
                 'borderColor': 'rgb(255, 99, 132)', 'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                 'borderDash': [5, 5], 'pointRadius': 0, 'showLine': True, 'fill': False, 'tension': 0.1}
            ]},
            'model_summary': final_models_summary,
            'selected_model_type': selected_model_name,
            'best_model_by_r2': best_model_name,
            'analysis_summary': {
                'input_data_points': len(df),
                'data_points_for_fit': len(train_df),
                'historical_oil_rate_stats': {
                    'min': float(train_df['OIL_RATE (stb/d)'].min()) if not train_df.empty else None,
                    'max': float(train_df['OIL_RATE (stb/d)'].max()) if not train_df.empty else None,
                    'mean': float(train_df['OIL_RATE (stb/d)'].mean()) if not train_df.empty else None,
                    'std_dev': float(train_df['OIL_RATE (stb/d)'].std()) if not train_df.empty else None
                },
                'historical_time_range_days': int(max_hist_rel_time) if not train_df.empty else 0,
                'forecast_period_days': forecast_days,
                'estimated_ultimate_recovery_30yr': float(eur_30_years) if np.isfinite(eur_30_years) else "Calculation Error or N/A",
                'production_date_range_analyzed': {
                    'start': initial_time.isoformat() if not df.empty else None,
                    'end': train_df['DATE'].max().isoformat() if not train_df.empty else None
                }
            },
            'chart_js_config_suggestion': { # Suggested Chart.js config
                'type': 'line',
                'data': {'datasets': 'USE_DATA_FROM_DATA_FOR_CHART_KEY'}, # Placeholder
                'options': {
                    'responsive': True, 'maintainAspectRatio': False,
                    'scales': {
                        'x': {'type': 'time', 'time': {'unit': 'month'}, 'title': {'display': True, 'text': 'Date'}},
                        'y': {'title': {'display': True, 'text': 'Oil Rate (stb/d)'}, 'beginAtZero': True}
                    },
                    'plugins': {
                        'title': {'display': True, 'text': f'Oil Production Decline Curve - Well: {well_id if well_id else "Combined"}'},
                        'legend': {'display': True, 'position': 'top'}
                    }
                }
            }
        }
        return response

    except Exception as e:
        tb_str = traceback.format_exc()
        # print(f"Error in API function: {tb_str}") # For server-side logs
        return {
            'success': False,
            'error': f'An unexpected error occurred in the API function: {str(e)}',
            'error_type': type(e).__name__,
            'traceback_snippet': tb_str.splitlines()[-5:] # Provide last few lines for context
        }

def get_well_score(oil, gas, water, artificial_score, workover_score, optimization_score):
    today = pd.to_datetime("today").normalize()
    current_date = today.strftime("%Y-%m-%d")

    # Fixed: Series max() doesn't need .iloc[0]
    oil_max = oil['OIL_RATE (stb/d)'].max()
    # Fixed: DataFrame filtering syntax
    oil_current_data = oil[oil['DATE'] == current_date]['OIL_RATE (stb/d)']
    if oil_current_data.empty:
        # Fallback to latest available data
        oil_current = oil.iloc[-1] if not oil.empty else 0
    else:
        oil_current = oil_current_data.iloc[0]
    oil_score = (oil_current / oil_max) * 100 if oil_max > 0 else 0

    gas_max = gas['GAS_RATE (MMscf/d)'].max()
    gas_current_data = gas[gas['DATE'] == current_date]['GAS_RATE (MMscf/d)']
    if gas_current_data.empty:
        gas_current = gas.iloc[-1] if not gas.empty else 0
    else:
        gas_current = gas_current_data.iloc[0]
    gas_score = (gas_current / gas_max) * 100 if gas_max > 0 else 0

    water_max = water['WATER_RATE (stb/d)'].max()
    water_current_data = water[water['DATE'] == current_date]['WATER_RATE (stb/d)']
    if water_current_data.empty:
        water_current = water.iloc[-1] if not water.empty else 0
    else:
        water_current = water_current_data.iloc[0]
    water_score = (water_current / water_max) * 100 if water_max > 0 else 0

    production = oil_score * 0.2 + gas_score * 0.1 - water_score * 0.1

    # Convert artificial lift color to score
    if artificial_score == "green":
        artificial_score_value = 100
    elif artificial_score == "yellow":
        artificial_score_value = 50
    elif artificial_score == "red":
        artificial_score_value = 0
    else:
        artificial_score_value = 100

    workover_score_value = workover_score * 100
    optimization_score_value = optimization_score * 100

    score = production + 0.2 * artificial_score_value + 0.3 * (100 - workover_score_value) + 0.2 * optimization_score_value
    return score

@fetch_api.route("/well-score", methods=["GET"])
def score_well():
    try:
        today = pd.to_datetime("today").normalize()
        current_date = today.strftime("%Y-%m-%d")
        project = client.get_default_project()
        well = request.args.get('well', None, type=str)
        area = well.split("-")[0]

        if not well:
            return jsonify({"error": "Well Id not defined"}), 400

        # Get workover data
        dataset = project.get_dataset("scored_MHI_distinct")
        workover_data = dataset.get_as_core_dataset()
        workover_data = workover_data.get_dataframe()

        # Get production data
        dataset = project.get_dataset("Well_data")
        production_data = dataset.get_as_core_dataset()
        production_data = production_data.get_dataframe()
        production_data['DATE'] = pd.to_datetime(production_data['DATE']).dt.strftime('%Y-%m-%d')

        # Fixed: dataset name from 'artlift_data' to 'airlift_data'
        dataset = project.get_dataset('artlift_data')
        airlift_data = dataset.get_as_core_dataset()
        airlift_data = airlift_data.get_dataframe()
        airlift_data['DATE'] = pd.to_datetime(airlift_data['DATE']).dt.strftime('%Y-%m-%d')

        score_well = {}
        wells = production_data["WELL"].unique().tolist()
        filtered_wells = [well for well in wells if area in well]
        for well in filtered_wells:
          production_data_filtered = production_data[production_data["WELL"] == well]
          if production_data_filtered.empty:
              return jsonify({"error": f"No production data found for well: {well}"}), 404

          gas_data = production_data_filtered[["DATE","GAS_RATE (MMscf/d)"]]
          oil_data = production_data_filtered[["DATE","OIL_RATE (stb/d)"]]
          water_data = production_data_filtered[["DATE","WATER_RATE (stb/d)"]]

          # Fixed: DataFrame filtering
          workover_filtered = workover_data[workover_data["WELL"] == well]
          if workover_filtered.empty:
              workover = 0
          else:
            workover = workover_filtered["workover_score"].iloc[0]

          # Fixed: DataFrame filtering syntax with proper boolean operators
          airlift_filtered = airlift_data[
              (airlift_data["WELL"] == well) &
              (airlift_data["DATE"] == current_date)
          ]
          if airlift_filtered.empty:
              return jsonify({"error": f"No airlift data found for well: {well} on date: {current_date}"}), 404
          airlift = airlift_filtered["Color"].iloc[0]

          # Randomize Optimzation Score
          optimization = random.uniform(0, 1)

          score = get_well_score(oil_data, gas_data, water_data, airlift, workover, optimization)
          well_detail = {}
          well_detail["score"] = score
          well_detail["wor"] = workover * 100
          score_well[well] = well_detail
        # Sort by score (descending - higher score = better rank)
        sorted_by_score = sorted(score_well.items(), key=lambda x: x[1]['score'], reverse=True)
        # Sort by WOR (ascending - lower WOR = better rank)
        sorted_by_wor = sorted(score_well.items(), key=lambda x: x[1]['wor'])

        # Add rankings
        for i, (well_name, data) in enumerate(sorted_by_score):
            score_well[well_name]['score_rank'] = i + 1

        for i, (well_name, data) in enumerate(sorted_by_wor):
            score_well[well_name]['wor_rank'] = i + 1

        score_well["area"] = area
        return jsonify(score_well)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@fetch_api.route("/artifical-monitoring", methods=["GET"])
def artificial_monitoring():
    try:
        today = pd.to_datetime("today").normalize()
        current_date = today.strftime("%Y-%m-%d")
        project = client.get_default_project()
        well = request.args.get('well', None, type=str)

        if not well:
            return jsonify({"error": "Well parameter is required"}), 400

        dataset = project.get_dataset('artlift_data')  # Fixed typo: 'artlift_data' -> 'airlift_data'
        airlift_data = dataset.get_as_core_dataset()
        airlift_data = airlift_data.get_dataframe()
        airlift_data['DATE'] = pd.to_datetime(airlift_data['DATE']).dt.strftime('%Y-%m-%d')

        # Convert numeric columns to proper data types
        numeric_columns = ['OIL_RATE (stb/d)', 'Vibration', 'Discharge Pressure', 'Intake Temperature', 'Motor Temperature']
        for col in numeric_columns:
            if col in airlift_data.columns:
                airlift_data[col] = pd.to_numeric(airlift_data[col], errors='coerce')

        # Fixed: Use & instead of and, and proper parentheses
        filtered_df = airlift_data[
            (airlift_data["WELL"] == well) &
            (airlift_data["Vibration"] != "-")
        ]

        line_chart = {}
        for i in range(30):
            date = (today - pd.DateOffset(days=i)).strftime("%Y-%m-%d")
            daily_data = filtered_df[filtered_df['DATE'] == date]

            if not daily_data.empty:
                line_chart[str(date)] = {
                    "oil": daily_data['OIL_RATE (stb/d)'].sum(),
                    "vibration": daily_data['Vibration'].mean(),
                    "pressure": daily_data['Discharge Pressure'].sum(),
                    "intake": daily_data['Intake Temperature'].mean(),
                    "motor": daily_data['Motor Temperature'].mean(),
                }
            else:
                line_chart[str(date)] = {
                    "oil": 0,
                    "vibration": 0,
                    "pressure": 0,
                    "intake": 0,
                    "motor": 0,
                }

        # Get current status
        current_status_data = filtered_df[filtered_df["DATE"] == current_date]

        # Fixed: Handle status properly
        if not current_status_data.empty:
            status = current_status_data["Color"].iloc[0]  # Get first value
        else:
            status = "unknown"  # Default status when no data available

        data = {
            "line_data": line_chart,
            "status": "Warning" if status == "yellow" else "Critical" if status == "red" else "Good" if status == "green" else status
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@fetch_api.route("/well-layer", methods=["GET"])
def well_layer():
    try:
      project = client.get_default_project()
      dataset = project.get_dataset('DatasetPerLayer')  # Fixed typo: 'artlift_data' -> 'airlift_data'
      layerData = dataset.get_as_core_dataset()
      layerData = layerData.get_dataframe()
      dataset = project.get_dataset('OverallInterventionRecommendations')  # Fixed typo: 'artlift_data' -> 'airlift_data'
      recomData = dataset.get_as_core_dataset()
      recomData = recomData.get_dataframe()
      list_well = [
          "NO 15/9-F-14 H",
          "NO 15/9-F-12 H",
          "NO 15/9-F-11 H"
      ]
      random_index = random.randint(0, 2)
      well = list_well[random_index]
      layerData = layerData[layerData['WELL_BORE_CODE'] == well]
      recomData = recomData[recomData['WELL_CODE'] == well]
      total_rate_well = layerData["CURRENT_WELL_OIL_RATE_BOPD"].iloc[0]
      list_layer = list(layerData['LAYER_ID'])
      data = {}
      data["total_rate_well"] = total_rate_well
      data["layers"] = []
      data["total_reservoir"] = 0
      data["incremental_prod"] = recomData['total_incremental_production_stbd'].iloc[0]
      data["workover_recom"] = recomData['intervention_mix'].iloc[0]
      for layer in list_layer:
          temp = {}
          temp["layer"] = layer
          curr_layer = layerData[layerData["LAYER_ID"] == layer].copy()
          temp["rate_layer"] = curr_layer["ALLOCATED_OIL_RATE_BOPD"].iloc[0]
          temp["total_reservoir"] = curr_layer["ORIGINAL_OIL_IN_PLACE_STB"].iloc[0]
          temp["prod_oil"] = curr_layer["CUMULATIVE_OIL_PRODUCED_STB"].iloc[0]
          temp["remaining_reservoir"] = curr_layer["REMAINING_RESERVES_STB"].iloc[0]
          data["total_reservoir"] =  data["total_reservoir"] + temp["remaining_reservoir"]
          data["layers"].append(temp)
      return jsonify(data)
    except Exception as e:
      return jsonify({"error": str(e)}), 500

@fetch_api.route("/map", methods=["GET"])
def get_map():
    try:
        project = client.get_default_project()
        dataset = project.get_dataset('ccus_world_data')
        mapData = dataset.get_as_core_dataset()
        mapData = mapData.get_dataframe()

        # Ensure numeric columns are numeric types
        mapData['latitude'] = pd.to_numeric(mapData['latitude'], errors='coerce')
        mapData['longitude'] = pd.to_numeric(mapData['longitude'], errors='coerce')
        mapData['code'] = pd.to_numeric(mapData['code'], errors='coerce')

        data = []
        for index, row in mapData.iterrows():
            temp = {
                'code': int(row['code']) if pd.notna(row['code']) else None,
                'region': str(row['region']) if pd.notna(row['region']) else None,
                'country': str(row['country']) if pd.notna(row['country']) else None,
                'area': str(row['area']) if pd.notna(row['area']) else None,
                'latitude': float(row['latitude']) if pd.notna(row['latitude']) else None,
                'longitude': float(row['longitude']) if pd.notna(row['longitude']) else None,
                'site_name': str(row['site_name']) if pd.notna(row['site_name']) else None,
                'basin': str(row['basin']) if pd.notna(row['basin']) else None,
                'unit_designation': str(row['unit_designation']) if pd.notna(row['unit_designation']) else None,
                'storage_unit_type': str(row['storage_unit_type']) if pd.notna(row['storage_unit_type']) else None,
                'rocktype': str(row['rocktype']) if pd.notna(row['rocktype']) else None,
                'project_spec': str(row['project_spec']) if pd.notna(row['project_spec']) else None,
                'co2': float(row['co2_density']) if pd.notna(row['co2_density']) else 0,
            }
            data.append(temp)

        return jsonify({
            "data": data
        })
    except Exception as e:
        print("Error: ", e)
        return jsonify({
            "error": str(e),
            "message": "Failed to fetch map data"
        }), 500

@fetch_api.route("/production-all",methods=["GET"])
def productionStatAll():
    try:
        # 1️⃣ Fetch dataset
        project = client.get_default_project()
        dataset = project.get_dataset("Well_data")
        df = dataset.get_as_core_dataset()
        df = df.get_dataframe()

        # DATE column is in 2025-05-26 00:00:00 format, we need to convert it to YYYY-MM-DD format
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
        # Filter out rows where DATE is NaN
        df = df.dropna(subset=['DATE'])

        # Extract AREA from WELL name (e.g., FAR-1 → FAR, DEA-1 → DEA)
        df['AREA'] = df['WELL'].str.split('-').str[0]

        today = pd.to_datetime("today").normalize()
        current_date = today.strftime("%Y-%m-%d")
        previous_date = (today - pd.DateOffset(days=1)).strftime("%Y-%m-%d")

        # 📊 METRIC: Sum from all areas for current and previous day
        current_data = df[df['DATE'] == current_date]
        previous_data = df[df['DATE'] == previous_date]

        # Group by area and calculate totals, then get sum across areas
        current_area_totals = current_data.groupby('AREA').agg({
            'GAS_RATE (MMscf/d)': 'sum',
            'WATER_RATE (stb/d)': 'sum',
            'OIL_RATE (stb/d)': 'sum',
            'DOWNHOLE_PRESSURE (psi)': 'mean'
        }).reset_index()

        previous_area_totals = previous_data.groupby('AREA').agg({
            'GAS_RATE (MMscf/d)': 'sum',
            'WATER_RATE (stb/d)': 'sum',
            'OIL_RATE (stb/d)': 'sum',
            'DOWNHOLE_PRESSURE (psi)': 'mean'
        }).reset_index()

        metric = {
            'current': {
                "gas": current_area_totals['GAS_RATE (MMscf/d)'].sum() if not current_area_totals.empty else 0,
                "water": current_area_totals['WATER_RATE (stb/d)'].sum() if not current_area_totals.empty else 0,
                "oil": current_area_totals['OIL_RATE (stb/d)'].sum() if not current_area_totals.empty else 0,
                "pressure": current_area_totals['DOWNHOLE_PRESSURE (psi)'].mean() if not current_area_totals.empty else 0,
            },
            'previous': {
                "gas": previous_area_totals['GAS_RATE (MMscf/d)'].sum() if not previous_area_totals.empty else 0,
                "water": previous_area_totals['WATER_RATE (stb/d)'].sum() if not previous_area_totals.empty else 0,
                "oil": previous_area_totals['OIL_RATE (stb/d)'].sum() if not previous_area_totals.empty else 0,
                "pressure": previous_area_totals['DOWNHOLE_PRESSURE (psi)'].mean() if not previous_area_totals.empty else 0,
            }
        }

        # 📈 LINE CHART: Separate data for gas, oil, water with 30 days history by AREA
        # Get all unique areas
        areas = df['AREA'].unique().tolist()

        # Generate date range for last 30 days
        date_range = []
        for i in range(29, -1, -1):  # 29 to 0 for chronological order
            date = (today - pd.DateOffset(days=i)).strftime("%Y-%m-%d")
            date_range.append(date)

        line_chart = {
            "gas": {
                "dates": date_range,
                "areas": {}
            },
            "oil": {
                "dates": date_range,
                "areas": {}
            },
            "water": {
                "dates": date_range,
                "areas": {}
            }
        }

        # Populate data for each area and each production type
        for area in areas:
            area_data = df[df['AREA'] == area]

            gas_data = []
            oil_data = []
            water_data = []

            for date in date_range:
                daily_area_data = area_data[area_data['DATE'] == date]

                if not daily_area_data.empty:
                    # Sum all wells in the area for that date
                    gas_data.append(daily_area_data['GAS_RATE (MMscf/d)'].sum())
                    oil_data.append(daily_area_data['OIL_RATE (stb/d)'].sum())
                    water_data.append(daily_area_data['WATER_RATE (stb/d)'].sum())
                else:
                    gas_data.append(0)
                    oil_data.append(0)
                    water_data.append(0)

            line_chart["gas"]["areas"][area] = gas_data
            line_chart["oil"]["areas"][area] = oil_data
            line_chart["water"]["areas"][area] = water_data

        # 🕸️ SPIDER CHART: Current production comparison across all areas
        spider_chart = {
            "areas": areas,
            "data": {
                "oil": [],
                "gas": [],
                "water": []
            }
        }

        for area in areas:
            area_current_data = df[(df['AREA'] == area) & (df['DATE'] == current_date)]

            if not area_current_data.empty:
                # Sum all wells in the area for current date
                spider_chart["data"]["oil"].append(area_current_data['OIL_RATE (stb/d)'].sum())
                spider_chart["data"]["gas"].append(area_current_data['GAS_RATE (MMscf/d)'].sum())
                spider_chart["data"]["water"].append(area_current_data['WATER_RATE (stb/d)'].sum())
            else:
                spider_chart["data"]["oil"].append(0)
                spider_chart["data"]["gas"].append(0)
                spider_chart["data"]["water"].append(0)

        # 📋 PRODUCTION SUMMARY
        # 1. First production date
        first_production_date = df['DATE'].min()

        # Get area max production for water target calculation
        area_max_production = df.groupby('AREA').agg({
            'OIL_RATE (stb/d)': 'max',
            'GAS_RATE (MMscf/d)': 'max',
            'WATER_RATE (stb/d)': 'max'
        })

        # 2. Average daily production (sum for today only)
        avg_daily_production = {
            "oil": current_area_totals['OIL_RATE (stb/d)'].sum() if not current_area_totals.empty else 0,
            "gas": current_area_totals['GAS_RATE (MMscf/d)'].sum() if not current_area_totals.empty else 0,
            "water": current_area_totals['WATER_RATE (stb/d)'].sum() if not current_area_totals.empty else 0
        }

        # 3. Target rate production (calculated based on averageDailyProduction percentages)
        target_rate = {
            "oil": avg_daily_production["oil"] / 0.90 if avg_daily_production["oil"] > 0 else 2200,  # averageDailyProduction should be 90% of target
            "gas": avg_daily_production["gas"] / 1.05 if avg_daily_production["gas"] > 0 else 10,    # averageDailyProduction should be 105% of target
            "water": area_max_production['WATER_RATE (stb/d)'].sum() * 0.8  # Keep existing calculation for water
        }

        # 4. Total production (cumulative by area, then summed)
        area_total_production = df.groupby('AREA').agg({
            'OIL_RATE (stb/d)': 'sum',
            'GAS_RATE (MMscf/d)': 'sum',
            'WATER_RATE (stb/d)': 'sum'
        })

        total_production = {
            "oil": area_total_production['OIL_RATE (stb/d)'].sum(),
            "gas": area_total_production['GAS_RATE (MMscf/d)'].sum(),
            "water": area_total_production['WATER_RATE (stb/d)'].sum()
        }

        # 5. Estimated total production (based on current trends - simplified calculation)
        # Use today's production rate for projection
        days_to_project = 365
        estimated_total_production = {
            "oil": total_production["oil"] + (avg_daily_production["oil"] * days_to_project),
            "gas": total_production["gas"] + (avg_daily_production["gas"] * days_to_project),
            "water": total_production["water"] + (avg_daily_production["water"] * days_to_project)
        }

        # 6. Reserve Analysis and Decline Rate Calculations
        # Decline rate assumptions (can be adjusted based on reservoir engineering)
        decline_rate_annual = 0.10  # 10% annual decline rate
        decline_rate_daily = decline_rate_annual / 365  # Daily decline rate

        # Calculate EUR (Estimated Ultimate Recovery) using decline curve analysis
        # Using simple exponential decline: EUR = Initial_Rate / Decline_Rate
        initial_oil_rate = df['OIL_RATE (stb/d)'].max() if not df.empty else 0
        initial_gas_rate = df['GAS_RATE (MMscf/d)'].max() if not df.empty else 0

        # EUR calculations (simplified using current production and decline rate)
        eur_oil_stb = (avg_daily_production["oil"] / decline_rate_daily) if decline_rate_daily > 0 and avg_daily_production["oil"] > 0 else 0
        eur_gas_mmscf = (avg_daily_production["gas"] / decline_rate_daily) if decline_rate_daily > 0 and avg_daily_production["gas"] > 0 else 0

        # Convert gas total production from MMscf to MMSCF for consistency
        total_production_gas_mmscf = total_production["gas"]

        # Calculate remaining reserves
        remaining_oil_stb = max(0, eur_oil_stb - total_production["oil"])
        remaining_gas_mmscf = max(0, eur_gas_mmscf - total_production_gas_mmscf)

        # Calculate remaining percentages
        remaining_oil_percentage = (remaining_oil_stb / eur_oil_stb * 100) if eur_oil_stb > 0 else 0
        remaining_gas_percentage = (remaining_gas_mmscf / eur_gas_mmscf * 100) if eur_gas_mmscf > 0 else 0

        # Reserve calculations
        reserves_analysis = {
            "oil": {
                "declineRateAnnual": decline_rate_annual,
                "declineRateDaily": decline_rate_daily,
                "estimatedEUR": eur_oil_stb,  # STB
                "totalProduced": total_production["oil"],  # STB
                "remainingReserves": remaining_oil_stb,  # STB
                "remainingPercentage": remaining_oil_percentage  # %
            },
            "gas": {
                "declineRateAnnual": decline_rate_annual,
                "declineRateDaily": decline_rate_daily,
                "estimatedEUR": eur_gas_mmscf,  # MMSCF
                "totalProduced": total_production_gas_mmscf,  # MMSCF
                "remainingReserves": remaining_gas_mmscf,  # MMSCF
                "remainingPercentage": remaining_gas_percentage  # %
            }
        }

        production_summary = {
            "firstProductionDate": first_production_date,
            "targetRate": target_rate,
            "totalProduction": total_production,
            "estimatedTotalProduction": estimated_total_production,
            "projectionDays": days_to_project,
            "averageDailyProduction": avg_daily_production,
            "reservesAnalysis": reserves_analysis
        }

        # Area breakdown for additional insights
        area_breakdown = {}
        for area in areas:
            area_data = df[df['AREA'] == area]
            area_current = area_data[area_data['DATE'] == current_date]
            area_wells = area_data['WELL'].unique().tolist()

            area_breakdown[area] = {
                "wells": area_wells,
                "wellCount": len(area_wells),
                "currentProduction": {
                    "oil": area_current['OIL_RATE (stb/d)'].sum() if not area_current.empty else 0,
                    "gas": area_current['GAS_RATE (MMscf/d)'].sum() if not area_current.empty else 0,
                    "water": area_current['WATER_RATE (stb/d)'].sum() if not area_current.empty else 0,
                },
                "totalProduction": {
                    "oil": area_data['OIL_RATE (stb/d)'].sum(),
                    "gas": area_data['GAS_RATE (MMscf/d)'].sum(),
                    "water": area_data['WATER_RATE (stb/d)'].sum()
                }
            }

        return jsonify({
            "metric": metric,
            "lineChart": line_chart,
            "spiderChart": spider_chart,
            "productionSummary": production_summary,
            "areaBreakdown": area_breakdown,
            "totalAreas": len(areas),
            "totalWells": len(df['WELL'].unique()),
            "dataDateRange": {
                "start": df['DATE'].min(),
                "end": df['DATE'].max()
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@fetch_api.route("/predict/ccus-trap", methods=["POST"])
def predict_ccus_trap():
    try:
        # Read JSON data from client
        client_data = request.get_json()

        # Validate required parameters
        required_params = ['type', 'latitude', 'region', 'formation', 'unit_designation', 'rocktype']
        for param in required_params:
            if param not in client_data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required parameter: {param}"
                }), 400

        model = loader.load_model('./models/ccstrap.zip')

        # Template data
        template_input = {
            "country": "Indonesia",
            "permeability_md": "159",
            "type": "Saline",
            "area": "Northern Territory: Offshore",
            "project_spec": False,
            "basin": "Bonaparte",
            "region": "Oceania",
            "rocktype": "Clastic",
            "unit_designation": "Saline Aquifer",
            "pre_injection_pressure": "0",
            "flowtest": False,
            "frac_pressure": "0",
            "formation": "Sandpiper Sandstone Formation",
            "source_of_storage_efficiency_factor": "Bradshaw et al 2009. Data from static and dynamic models used to constrain inputs (variability evaluated using Monte Carlo model to address uncertainties). Probabilistic effective storage estimate.",
            "pore_compressibility": "0",
            "brine_salinity": "0",
            "depth": "2200",
            "age": "Lower Cretaceous",
            "pressure_psig": 0.0,
            "longitude": 129.74277777777777,
            "co2_density": 0.0,
            "well_density": 0.00425,
            "thickness_m": 168.5,
            "site_area_km2": 8000.0,
            "latitude": -13.227500000000001,
            "single_well_discovery_area": 200,
            "well_count": 34,
            "ntg": 0.97,
            "porosity": 19.7,
            "storage_unit_type": None,
            "code": None,
            "site_name": None,
            "discovery_status": None,
            "publication": None,
            "project_history": None,
            "development_plan": None,
            "containment_summary": None,
            "assessment_notes": None,
            "year_of_publication": None,
            "date_of_assessment": None,
            "source_of_analogue": None,
            "assessment": None,
            "stored_low": None,
            "stored_mid": None,
            "stored_high": None,
            "on_injection_low": None,
            "on_injection_mid": None,
            "on_injection_high": None,
            "approved_for_development_low": None,
            "approved_for_development_mid": None,
            "approved_for_development_high": None,
            "justified_for_development_low": None,
            "justified_for_development_mid": None,
            "justified_for_development_high": None,
            "development_pending_low": None,
            "development_pending_mid": None,
            "development_pending_high": None,
            "development_on_hold_low": None,
            "development_on_hold_mid": None,
            "development_on_hold_high": None,
            "development_not_viable_low": None,
            "development_not_viable_mid": None,
            "development_not_viable_high": None,
            "development_unclarified_low": None,
            "development_unclarified_mid": None,
            "development_unclarified_high": None,
            "inaccessible_subcommercial_low": None,
            "inaccessible_subcommercial_mid": None,
            "inaccessible_subcommercial_high": None,
            "prospect_low": None,
            "prospect_mid": None,
            "prospect_high": None,
            "lead_low": None,
            "lead_mid": None,
            "lead_high": None,
            "sequence_play_low": None,
            "sequence_play_mid": None,
            "sequence_play_high": None,
            "basin_play_low": None,
            "basin_play_mid": None,
            "basin_play_high": None,
            "undiscovered_inaccessible_low": None,
            "undiscovered_inaccessible_mid": None,
            "undiscovered_inaccessible_high": None,
            "total_low": None,
            "total_mid": None,
            "total_high": None,
            "sum_low": None,
            "sum_mid": None,
            "sum_high": None,
            "p50_pore_volume_mmcum": None,
            "prop_considered_discovered": None,
            "storage_efficiency": None
        }

        # Override template_input with client_data if available
        for key in template_input:
            if key in client_data:
                template_input[key] = client_data[key]

        df = pd.DataFrame([template_input])
        prediction = model.predict(df)

        return jsonify({
            "success": True,
            "prediction": prediction[0],
            "model_info": {
                "id": "3qgVTU8C",
                "name": "CCUS Trap Prediction",
                "type": "PREDICTION"
            }
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to predict CCUS trap"
        }), 500


@fetch_api.route("/predict/ccus-go-nogo", methods=["POST"])
def predict_ccus_go_nogo():
    try:
        # Read JSON data from client
        client_data = request.get_json()

        # Validate required parameters
        required_params = ['formation', 'depth', 'site_area_km2', 'frac_pressure', 'country', 'unit_designation']
        for param in required_params:
            if param not in client_data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required parameter: {param}"
                }), 400

        # Map formation type
        formation_map = {
            "Sandstone": "Sandpiper Sandstone Formation",
            "Carbonate": "Carbonate Formation"
        }
        formation = formation_map.get(client_data['formation'], "Sandpiper Sandstone Formation")

        # Map unit designation
        unit_designation_map = {
            "Saline": "Saline Aquifer",
            "Oil & Gas Fields": "Depleted Oil & Gas Field",
            "Oilfield": "Depleted Oil & Gas Field"
        }
        unit_designation = unit_designation_map.get(client_data['unit_designation'], "Saline Aquifer")

        # Map type based on unit_designation
        type_map = {
            "Saline Aquifer": "Saline",
            "Depleted Oil & Gas Field": "Depleted"
        }
        storage_type = type_map.get(unit_designation, "Saline")

        # Get reasonable rocktype based on formation
        rocktype = "Clastic" if client_data['formation'] == "Sandstone" else "Carbonate"

        # Indonesia-specific reasonable values
        # Common Indonesian basins: Bonaparte, Java Sea, Kutei, South Sumatra, etc.
        basin = "Java Sea"  # atau "Bonaparte", "Kutei", "South Sumatra"
        area = "Java Sea: Offshore"  # sesuaikan dengan basin

        # Indonesia coordinates (Jakarta area as default)
        latitude = -6.2088  # Jakarta latitude
        longitude = 106.8456  # Jakarta longitude

        model = loader.load_model('./models/gonogo.zip')

        template_input = {
            # INPUT FEATURES - Categorical (Dummy encoding)
            "country": client_data['country'],
            "permeability_md": "159",  # Reasonable value for Indonesian formations
            "type": storage_type,
            "area": area,
            "project_spec": False,
            "basin": basin,
            "region": "Oceania",  # Indonesia is in Oceania region
            "rocktype": rocktype,
            "unit_designation": unit_designation,
            "pre_injection_pressure": "0",
            "flowtest": False,
            "frac_pressure": str(client_data['frac_pressure']),  # Convert from bar
            "formation": formation,
            "source_of_storage_efficiency_factor": "Regional assessment based on Indonesian geological data",
            "pore_compressibility": "0",
            "brine_salinity": "35000",  # Typical salinity for Indonesian offshore (ppm)
            "depth": str(client_data['depth']),  # Already in ft from client
            "age": "Miocene",  # Common age for Indonesian oil/gas formations

            # INPUT FEATURES - Numerical (Avg-std rescaling)
            "pressure_psig": client_data['depth'] * 0.433,  # Hydrostatic pressure estimate (0.433 psi/ft)
            "longitude": longitude,
            "co2_density": 0.65,  # Typical CO2 density at Indonesian reservoir conditions
            "well_density": 0.005,  # Reasonable for Indonesian fields
            "thickness_m": 100.0,  # Reasonable thickness for Indonesian reservoirs
            "site_area_km2": client_data['site_area_km2'],
            "latitude": latitude,
            "single_well_discovery_area": 150,  # Reasonable for Indonesian fields
            "well_count": int(client_data['site_area_km2'] * 0.005 * 4),  # Estimate based on well density
            "ntg": 0.85,  # Net to gross ratio typical for Indonesian clastic reservoirs
            "porosity": 20.0 if rocktype == "Clastic" else 15.0,  # Typical porosity values

            # TARGET FEATURE
            "storage_unit_type": None,

            # REJECTED FEATURES - Set to None
            "code": None,
            "site_name": None,
            "discovery_status": None,
            "publication": None,
            "project_history": None,
            "development_plan": None,
            "containment_summary": None,
            "assessment_notes": None,
            "year_of_publication": None,
            "date_of_assessment": None,
            "source_of_analogue": None,
            "assessment": None,
            "stored_low": None,
            "stored_mid": None,
            "stored_high": None,
            "on_injection_low": None,
            "on_injection_mid": None,
            "on_injection_high": None,
            "approved_for_development_low": None,
            "approved_for_development_mid": None,
            "approved_for_development_high": None,
            "justified_for_development_low": None,
            "justified_for_development_mid": None,
            "justified_for_development_high": None,
            "development_pending_low": None,
            "development_pending_mid": None,
            "development_pending_high": None,
            "development_on_hold_low": None,
            "development_on_hold_mid": None,
            "development_on_hold_high": None,
            "development_not_viable_low": None,
            "development_not_viable_mid": None,
            "development_not_viable_high": None,
            "development_unclarified_low": None,
            "development_unclarified_mid": None,
            "development_unclarified_high": None,
            "inaccessible_subcommercial_low": None,
            "inaccessible_subcommercial_mid": None,
            "inaccessible_subcommercial_high": None,
            "prospect_low": None,
            "prospect_mid": None,
            "prospect_high": None,
            "lead_low": None,
            "lead_mid": None,
            "lead_high": None,
            "sequence_play_low": None,
            "sequence_play_mid": None,
            "sequence_play_high": None,
            "basin_play_low": None,
            "basin_play_mid": None,
            "basin_play_high": None,
            "undiscovered_inaccessible_low": None,
            "undiscovered_inaccessible_mid": None,
            "undiscovered_inaccessible_high": None,
            "total_low": None,
            "total_mid": None,
            "total_high": None,
            "sum_low": None,
            "sum_mid": None,
            "sum_high": None,
            "p50_pore_volume_mmcum": None,
            "prop_considered_discovered": None,
            "storage_efficiency": None
        }

        df = pd.DataFrame([template_input])
        prediction = model.predict(df)

        return jsonify({
            "success": True,
            "prediction": prediction[0],
            "model_info": {
                "id": "3qgVTU8C",
                "name": "Predict Go/Nogo - CCUS",
                "type": "PREDICTION"
            }
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to predict CCUS Go/Nogo"
        }), 500


    project = client.get_default_project()
    try:
        # 1. Ambil data JSON dari request
        input_data = request.get_json()
        if not input_data:
            return jsonify({
                "success": False,
                "error": "Invalid input",
                "message": "Request body harus berisi JSON."
            }), 400

        # 2. Load semua 3 model untuk DCA parameters
        model_qi_obj = project.get_saved_model('obNojUbY')  # DCA_QI
        model_di_obj = project.get_saved_model('IjlqFq4w')  # DCA_DI
        model_b_obj = project.get_saved_model('mh8CKwbK')   # DCA_B

        # 3. Download dan load model untuk mendapatkan feature requirements
        # Get active version untuk model pertama (asumsi semua model punya feature yang sama)
        active_version_id = model_qi_obj.get_active_version().get('id')
        version_details = model_qi_obj.get_version_details(active_version_id)
        version_details.get_scoring_python("./model_qi.zip")

        # Load model QI untuk mendapatkan feature names
        model_qi = loader.load_model(export_path="./model_qi.zip")

        # Dapatkan semua feature names yang dibutuhkan
        raw_details = version_details.get_raw()
        all_feature_names = list(raw_details['preprocessing']['per_feature'].keys())
        print(f"All features needed: {all_feature_names}")

        # Download model lainnya
        active_version_di = model_di_obj.get_active_version().get('id')
        version_details_di = model_di_obj.get_version_details(active_version_di)
        version_details_di.get_scoring_python("./model_di.zip")
        model_di = loader.load_model(export_path="./model_di.zip")

        active_version_b = model_b_obj.get_active_version().get('id')
        version_details_b = model_b_obj.get_version_details(active_version_b)
        version_details_b.get_scoring_python("./model_b.zip")
        model_b = loader.load_model(export_path="./model_b.zip")

        # 4. Konversi JSON ke Pandas DataFrame dengan semua kolom yang dibutuhkan
        input_df = pd.DataFrame(columns=all_feature_names, index=[0])
        input_df.update(pd.Series(input_data))

        # 5. Lakukan prediksi untuk ketiga parameter DCA
        # Prediksi QI (Initial Production Rate)
        qi_result = model_qi.predict(input_df)
        qi_prediction = qi_result['prediction'][0] if 'prediction' in qi_result else qi_result[0]

        # Prediksi DI (Initial Decline Rate)
        di_result = model_di.predict(input_df)
        di_prediction = di_result['prediction'][0] if 'prediction' in di_result else di_result[0]

        # Prediksi B (Decline Exponent)
        b_result = model_b.predict(input_df)
        b_prediction = b_result['prediction'][0] if 'prediction' in b_result else b_result[0]

        print(f"\nPrediction Results:")
        print(f"QI (Initial Rate): {qi_prediction}")
        print(f"DI (Decline Rate): {di_prediction}")
        print(f"B (Decline Exponent): {b_prediction}")

        # 6. Siapkan response dalam format JSON
        return jsonify({
            "success": True,
            "predictions": {
                "qi": float(qi_prediction),  # Initial Production Rate
                "di": float(di_prediction),  # Initial Decline Rate
                "b": float(b_prediction)     # Decline Exponent
            },
            "model_info": {
                "qi_model": "DCA_QI Predictor",
                "di_model": "DCA_DI Predictor",
                "b_model": "DCA_B Predictor"
            },
            "features_used": len(all_feature_names),
            "required_features": [
                'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_DOWNHOLE_PRESSURE',
                'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_WHP_P', 'AVG_WHT_P',
                'CUM_OIL_VOL_WELL', 'DP_CHOKE_SIZE', 'SYN_GOR_SCF_STB',
                'SYN_OIL_DENSITY_API', 'SYN_OIL_FVF_PROXY_RB_STB', 'SYN_PERFORATION_EFFICIENCY',
                'SYN_RESERVOIR_PRESSURE_PROXY_PSIA', 'SYN_SEG_AVG_INITIAL_SW',
                'SYN_SEG_AVG_PERM_MD', 'SYN_SEG_AVG_POROSITY', 'SYN_SEG_EFFECTIVE_KH_PROXY_MD_FT',
                'SYN_SEG_MOBILITY_PROXY', 'SYN_SEG_NET_PAY_FT', 'SYN_SEG_OOIP_PROXY_MMSTB',
                'SYN_SEG_RE_FT', 'SYN_SKIN_FACTOR', 'SYN_WATER_INJECTION_SUPPORT_FACTOR'
            ]
        })

    except KeyError as e:
        print(f"Error pada saat prediksi: Kunci tidak ditemukan. Error: {e}")
        return jsonify({
            "success": False,
            "error": f"Key not found: {e}. Check model details or prediction output.",
            "message": "Gagal memproses prediksi. Kunci yang dibutuhkan tidak ditemukan."
        }), 500

    except Exception as e:
        print(f"Error pada saat prediksi: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Gagal melakukan prediksi DCA. Pastikan format input data sudah benar dan semua fitur yang dibutuhkan tersedia."
        }), 500

@fetch_api.route("/predict/ccus-eor", methods=["POST"])
def predict_ccus_eor():
    try:
        # Read JSON data from client
        client_data = request.get_json()
        # Validate required parameters
        required_params = ['porosity', 'permeability', 'depth', 'oil_gravity', 'oil_viscosity', 'formation']
        for param in required_params:
            if param not in client_data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required parameter: {param}"
                }), 400

        model = loader.load_model('./models/ccuseor.zip')
        if client_data['formation'] == "Sandstone":
            client_data['formation'] = "S"
        elif client_data['formation'] == "Carbonate":
            client_data['formation'] = "C"

        template_input = {
            # 🟦 Categorical - Dummy encoding
            "operator": "Berry",
            "Field": "South Midway–Sunset",
            "State": "Calif.",
            "County": "Kern",
            "Start Date": "1964",
            "Pay zone": "Monarch",
            "Formation": client_data['formation'],
            # 🟩 Numerical - Avg-std rescaling
            "Area, acres": 600,
            "Total Wells prod.": 1200,
            "Total Wells Inj.": None,  # Empty → None
            "Porosity": client_data['porosity'],
            "Permeabiliy, mD": client_data['permeability'],  # Fixed typo here
            "Depth, ft": client_data['depth'],
            "Oil Gravity, API": client_data['oil_gravity'],
            "Oil Viscosity, cP": client_data['oil_viscosity'],
            "Temperature, F": 80,
            "Total Prod, b/d": 10000,
            "Enhanced Prod. b/d": 7000,
            # ❌ Rejected columns and non-inputs
            "EOR Type": None,
            "col_19": None,
            "col_20": None,
            "col_21": None,
            "col_22": None,
            "col_23": None,
            "col_24": None,
            "col_25": None,
            "EOR_Type_Declutter": None
        }

        df = pd.DataFrame([template_input])
        prediction = model.predict(df)
        return jsonify({
            "success": True,
            "prediction": prediction[0],
            "model_info": {
                "id": "Zby5vsSm",
                "name": "Predict EOR - CCUS Study",
                "type": "PREDICTION"
            }
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to predict CCUS EOR"
        }), 500

@fetch_api.route("/predict/esp-failure", methods=["GET"])
def predict_esp_failure_endpoint():
    project = client.get_default_project()
    try:
        # 1. Terima request
        well_name = request.args.get('well', None)
        if not well_name:
            return jsonify({"success": False, "error": "Query parameter 'well' is required."}), 400

        print(f"API dipanggil untuk memicu prediksi sumur: {well_name}")

        # 2. Set Project Variable untuk memberitahu Recipe
        project_vars = project.get_variables()
        project_vars['standard']['api_well_name'] = well_name
        project.set_variables(project_vars)

        # 3. Jalankan Scenario
        # GANTI DENGAN ID SKENARIO ANDA
        scenario = project.get_scenario('API_Prediction_Results')
        run = scenario.run_and_wait() # run_and_wait() lebih simpel dari run() + wait_for_completion()

        # Cek apakah skenario berhasil
        outcome = run.get_info()['result']['outcome']
        if outcome != 'SUCCESS':
            raise Exception(f"Skenario Dataiku gagal dengan status: {outcome}")

        print("Skenario berhasil. Mengambil hasil...")

        # 4. Ambil hasil dari dataset output

        results_dataset = project.get_dataset("API_Prediction_Results")
        results_df = results_dataset.get_as_core_dataset().get_dataframe()

        well_result = results_df[results_df['well_name'] == well_name]
        if well_result.empty:
            raise Exception("Skenario berhasil, tapi tidak ada hasil ditemukan untuk sumur ini.")

        latest_result = well_result.sort_values(by='prediction_timestamp_utc', ascending=False).iloc[0]

        PRESCRIPTIVE_ACTIONS = {
            "Low PI": {
                "priority": "High",
                "primary_action": "Investigate potential inflow restriction.",
                "steps": [
                    "Check current fluid level and calculate Bottom Hole Pressure (BHP).",
                    "Compare current BHP with historical data to confirm productivity decline.",
                    "If pump design allows, consider adjusting tubing head pressure to match lower inflow."
                ],
                "possible_causes": "Scale buildup, paraffin deposition, or reservoir pressure depletion."
            },
            "Tubing Leak": {
                "priority": "Critical",
                "primary_action": "Confirm integrity of the production tubing immediately.",
                "steps": [
                    "Initiate a wellhead pressure test to check for leaks.",
                    "If pressure test is inconclusive, perform a dead-head test.",
                    "Prepare for potential well intervention if a leak is confirmed."
                ],
                "possible_causes": "Corrosion, connection failure, or mechanical damage."
            },
            "Pump Wear": {
                "priority": "Medium",
                "primary_action": "Evaluate pump performance degradation.",
                "steps": [
                    "Analyze system efficiency trends (e.g., energy consumption per barrel lifted).",
                    "Review vibration data for sustained increases.",
                    "Begin planning for a future pump replacement (workover)."
                ],
                "possible_causes": "Abrasive wear, corrosion, or operating outside of design range."
            },
            "Sand Ingestion": {
                "priority": "High",
                "primary_action": "Mitigate solids production to prevent catastrophic pump failure.",
                "steps": [
                    "Immediately check surface equipment for sand accumulation.",
                    "Consider reducing flow rate to minimize solids lifting.",
                    "Evaluate need for downhole sand control for next workover."
                ],
                "possible_causes": "Formation failure, high drawdown, or ineffective sand control."
            },
            "Closed Valve (SSSV)": {
                "priority": "High",
                "primary_action": "Verify status of downhole and surface safety valves.",
                "steps": [
                    "Confirm the intended position of the SSSV and surface valves.",
                    "If unintentional, contact Field Service Tech for immediate on-site inspection.",
                    "Do not restart pump until valve status is confirmed."
                ],
                "possible_causes": "Accidental closure, control line failure, or safety shutdown."
            },
            "Increase in Frequency": {
                "priority": "Low",
                "primary_action": "Review recent VSD (Variable Speed Drive) frequency changes.",
                "steps": [
                    "Verify if frequency increase was an intentional operational change.",
                    "Monitor intake pressure closely to avoid pump-off condition.",
                    "If unintentional, revert to the previous setpoint and monitor."
                ],
                "possible_causes": "Manual operator adjustment or automated optimization."
            },
            "Normal Operation": {
                "priority": "Info",
                "primary_action": "System operating within expected parameters.",
                "steps": [
                    "Continue routine monitoring.",
                    "No immediate intervention required."
                ],
                "possible_causes": "N/A"
            },
            "default": {
                "priority": "High",
                "primary_action": "Unrecognized pattern detected. Manual analysis required.",
                "steps": [
                    "Review recent trends of all sensor data.",
                    "Compare current operations with the daily well plan.",
                    "Alert senior production engineer for investigation."
                ],
                "possible_causes": "Complex failure mode, sensor malfunction, or a new, unlearned pattern."
            }
        }

        # 5. Ekstrak semua data dan bangun response JSON yang kaya
        final_prediction = latest_result['failure_mode']
        prediction_proba = latest_result['confidence']

        # Dapatkan detail aksi dari kamus
        action_details = PRESCRIPTIVE_ACTIONS.get(final_prediction, PRESCRIPTIVE_ACTIONS["default"])

        # Buat response akhir
        return jsonify({
            "success": True,
            "well_name": latest_result['well_name'],
            "prediction_timestamp_utc": latest_result['prediction_timestamp_utc'].isoformat(),
            "prediction_details": {
                "failure_mode": final_prediction,
                "confidence": round(float(prediction_proba), 4),
                "priority": action_details.get("priority", "Unknown")
            },
            "recommended_action": {
                "summary": action_details.get("primary_action", "No action defined."),
                "detailed_steps": action_details.get("steps", []),
                "possible_causes": action_details.get("possible_causes", [])
            },
            "contributing_factors": {
                "Intake_Pressure_Trend": latest_result['Intake_Pressure_trend'],
                "Discharge_Pressure_Trend": latest_result['Discharge_Pressure_trend'],
                "Vibration_Trend": latest_result['Vibration_trend'],
                "Oil_Rate_Trend": latest_result['OIL_RATE_numeric_trend']
            }
        })

    except Exception as e:
        print(f"FATAL ERROR di endpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred.",
            "details": str(e)
        }), 500


@fetch_api.route("/predict/dca-parameters", methods=["POST"])
def predict_dca_parameters():
    project = client.get_default_project()
    try:
        # 1. Ambil data JSON dari request
        input_data = request.get_json()
        if not input_data:
            return jsonify({
                "success": False,
                "error": "Invalid input",
                "message": "Request body harus berisi JSON."
            }), 400

        # 2. Load semua 3 model untuk DCA parameters
        model_qi_obj = project.get_saved_model('obNojUbY')  # DCA_QI
        model_di_obj = project.get_saved_model('IjlqFq4w')  # DCA_DI
        model_b_obj = project.get_saved_model('mh8CKwbK')   # DCA_B

        # 3. Download dan load model untuk mendapatkan feature requirements
        # Get active version untuk model pertama (asumsi semua model punya feature yang sama)
        active_version_id = model_qi_obj.get_active_version().get('id')
        version_details = model_qi_obj.get_version_details(active_version_id)
        version_details.get_scoring_python("./model_qi.zip")

        # Load model QI untuk mendapatkan feature names
        model_qi = loader.load_model(export_path="./model_qi.zip")

        # Dapatkan semua feature names yang dibutuhkan
        raw_details = version_details.get_raw()
        all_feature_names = list(raw_details['preprocessing']['per_feature'].keys())
        print(f"All features needed: {all_feature_names}")

        # Download model lainnya
        active_version_di = model_di_obj.get_active_version().get('id')
        version_details_di = model_di_obj.get_version_details(active_version_di)
        version_details_di.get_scoring_python("./model_di.zip")
        model_di = loader.load_model(export_path="./model_di.zip")

        active_version_b = model_b_obj.get_active_version().get('id')
        version_details_b = model_b_obj.get_version_details(active_version_b)
        version_details_b.get_scoring_python("./model_b.zip")
        model_b = loader.load_model(export_path="./model_b.zip")

        # 4. Konversi JSON ke Pandas DataFrame dengan semua kolom yang dibutuhkan
        input_df = pd.DataFrame(columns=all_feature_names, index=[0])
        input_df.update(pd.Series(input_data))

        # 5. Lakukan prediksi untuk ketiga parameter DCA
        # Prediksi QI (Initial Production Rate)
        qi_result = model_qi.predict(input_df)
        qi_prediction = qi_result['prediction'][0] if 'prediction' in qi_result else qi_result[0]

        # Prediksi DI (Initial Decline Rate)
        di_result = model_di.predict(input_df)
        di_prediction = di_result['prediction'][0] if 'prediction' in di_result else di_result[0]

        # Prediksi B (Decline Exponent)
        b_result = model_b.predict(input_df)
        b_prediction = b_result['prediction'][0] if 'prediction' in b_result else b_result[0]

        print(f"\nPrediction Results:")
        print(f"QI (Initial Rate): {qi_prediction}")
        print(f"DI (Decline Rate): {di_prediction}")
        print(f"B (Decline Exponent): {b_prediction}")

        # 6. Siapkan response dalam format JSON
        return jsonify({
            "success": True,
            "predictions": {
                "qi": float(qi_prediction),  # Initial Production Rate
                "di": float(di_prediction),  # Initial Decline Rate
                "b": float(b_prediction)     # Decline Exponent
            },
            "model_info": {
                "qi_model": "DCA_QI Predictor",
                "di_model": "DCA_DI Predictor",
                "b_model": "DCA_B Predictor"
            },
            "features_used": len(all_feature_names),
            "required_features": [
                'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_DOWNHOLE_PRESSURE',
                'AVG_DOWNHOLE_TEMPERATURE', 'AVG_DP_TUBING', 'AVG_WHP_P', 'AVG_WHT_P',
                'CUM_OIL_VOL_WELL', 'DP_CHOKE_SIZE', 'SYN_GOR_SCF_STB',
                'SYN_OIL_DENSITY_API', 'SYN_OIL_FVF_PROXY_RB_STB', 'SYN_PERFORATION_EFFICIENCY',
                'SYN_RESERVOIR_PRESSURE_PROXY_PSIA', 'SYN_SEG_AVG_INITIAL_SW',
                'SYN_SEG_AVG_PERM_MD', 'SYN_SEG_AVG_POROSITY', 'SYN_SEG_EFFECTIVE_KH_PROXY_MD_FT',
                'SYN_SEG_MOBILITY_PROXY', 'SYN_SEG_NET_PAY_FT', 'SYN_SEG_OOIP_PROXY_MMSTB',
                'SYN_SEG_RE_FT', 'SYN_SKIN_FACTOR', 'SYN_WATER_INJECTION_SUPPORT_FACTOR'
            ]
        })

    except KeyError as e:
        print(f"Error pada saat prediksi: Kunci tidak ditemukan. Error: {e}")
        return jsonify({
            "success": False,
            "error": f"Key not found: {e}. Check model details or prediction output.",
            "message": "Gagal memproses prediksi. Kunci yang dibutuhkan tidak ditemukan."
        }), 500

    except Exception as e:
        print(f"Error pada saat prediksi: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Gagal melakukan prediksi DCA. Pastikan format input data sudah benar dan semua fitur yang dibutuhkan tersedia."
        }), 500
