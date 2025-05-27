from flask import Blueprint, jsonify, request
from client import Dataiku
from dotenv import load_dotenv
import pandas as pd
import random

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

