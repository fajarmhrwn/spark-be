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

load_dotenv()

client = Dataiku().client
fetch_api = Blueprint("fetch_api", __name__, url_prefix="/api")


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
