import requests
# imports:
import pandas as pd
import re
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import json

def prepare_data(data):


    # הפונקציות שאשתמש בהן בהמשך
    def convert_to_binary(x):
        x = str(x)
        if (re.search("^יש", x) is not None) or (re.search("^כן", x) is not None) or (x == "yes") or (x == "True") or (
                re.search("^נגיש", x) is not None):
            return 1
        elif (re.search("^משופץ", x) is not None) or (re.search("^חדש", x) is not None) or (
                re.search("^שמור", x) is not None):
            return None
        else:
            return 0

    # ---------------------------------------------------------------------------------------------------------------------------
    def to_room_num(x):
        if re.findall('[0-9]+.[0-9]+', str(x)):
            return float(re.findall('[0-9]+.[0-9]+', str(x))[0])
        if re.findall('[0-9]+', str(x)):
            return float(re.findall('[0-9]+', str(x))[0])
        elif isinstance(x, list):
            if re.findall('[0-9]+.[0-9]+', str(x[0])):
                return float(re.findall('[0-9]+.[0-9]+', str(x[0])))
            else:
                return float(re.findall('[0-9]+', str(x))[0])
        else:
            return None

    # ---------------------------------------------------------------------------------------------------------------------------
    def to_other_type(x):
        if x == "בניין" or x == "מיני פנטהאוז" or x == "מגרש" or x == "קוטג' טורי" or x == "דירת נופש" or x == "אחר" or x == "טריפלקס" or x == "נחלה":
            return "אחר"
        elif re.findall('קוטג', str(x)):
            return "קוטג"
        else:
            return x

    # ---------------------------------------------------------------------------------------------------------------------------
    def calculate_entrance_date(date):
        if re.findall('מיידי', str(date)):
            return "less_than_6 months"

        elif re.findall('גמיש', str(date)):
            return "flexible"
        elif re.findall('לא צויין', str(date)):
            return "not_defined"
        else:

            date = pd.to_datetime(date)
            #         today = date.today()
            today = pd.to_datetime('today')
            date_difference = date - today

            days_difference = date_difference.days
            if days_difference < 182:
                return 'less_than_6 months'
            elif days_difference > 182 and days_difference < 365:
                return "months_6_12"
            else:
                return "above_year"

    # ---------------------------------------------------------------------------------------------------------------------------

    def to_condition(x):
        if x == "מיידי" or x == "גמיש" or x == False or x == "לא צויין" or x == 'None' or x is np.nan:
            return "לא צויין"
        elif x == "ישן" or x == "דורש שיפוץ":
            return "ישן"
        elif re.findall('חדש', str(x)):
            return "חדש"
        else:
            return x

    def total_floor(x):
        if re.findall('[0-9]+', str(x)):
            if len(re.findall('[0-9]+', str(x))) == 2:
                if float(re.findall('[0-9]+', str(x))[1]) == 0:
                    return float(re.findall('[0-9]+', str(x))[0])
                else:
                    return float(re.findall('[0-9]+', str(x))[1])
            elif len(re.findall('[0-9]+', str(x))) == 1:
                return float(re.findall('[0-9]+', str(x))[0])
        elif re.findall('קרקע', str(x)) or x is np.nan or re.findall('מרתף', str(x)):
            return 1

    def to_floor(x):
        if re.findall('[0-9]+', str(x)):
            return int(re.findall('[0-9]+', str(x))[0])
        elif re.findall('קרקע', str(x)) or re.findall('מרתף', str(x)) or x is np.nan:
            return 0
        else:
            return 0

    def to_furniture(x):
        if re.findall('ללא', str(x)):
            return "אין"
        else:
            return x

  

    # ____________________________________________________________________________________________________
    def check(url):
        try:
            response = requests.get(url)
            if not response.status_code == 200:
                print("HTTP error", response.status_code)
            else:
                try:
                    response_data = response.json()
                except:
                    print("Response not in valid JSON format")
        except:
            print("Something went wrong with requests.get")
        return response_data

    # ____________________________________________________________________________________________________
    def get_region(destination_address):
        api_key = 'AIzaSyCUcFSaymLrKYtbPMpYXO16I93VkI2mjvg'
        if re.findall('אביב', destination_address):
            origin_address = "יפו ג תל אביב"
        else:
            origin_address = "אילת"
            
        url = f'https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin_address}&destinations={destination_address}&key={api_key}'
        data1 = check(url)
        distance = data1["rows"][0]["elements"][0]['distance']['text']
        #     print(distance,destination_address)
        if re.findall('([0-9]+) km', distance):
            distance = re.findall('([0-9]+) km', distance)[0]
        elif re.findall('([0-9]+) m', distance):
            distance = re.findall('([0-9]+) m', distance)[0]
        distance = float(distance.replace(",", ""))

        if origin_address == "יפו ג תל אביב":
            if distance <= 7:
                return "תל אביב-זול"
            else:
                return "תל אביב-יקר"

        else:
            if distance < 300:
                return "דרום"
            elif distance >= 300 and distance < 380:
                return "מרכז"
            else:
                return "צפון"

    # השינויים עצמם בדאטה
    def combine_columns(row):
        if row['City'] == 'תל אביב' :
            return str(row['City']) + ' ' + str(row['city_area'])
        else:
            # print(row)
            return str(row['City'])

    def encode_columns(data):
        data1 = data.copy()
        data1 = data1.drop(['City', 'city_area', 'City_Combined', 'type', 'condition ', 'furniture ', 'entranceDate '],
                           axis=1)

        one_hot_encoder = OneHotEncoder()
        type_order =['אחר',"דירה","דופלקס","דירת גן","דירת גג","פנטהאוז","קוטג","דו משפחתי","בית פרטי"]
        furniture_order = ['לא צויין', 'אין', 'חלקי', 'מלא']
        entranceDate_order = ['flexible', "less_than_6 months", 'months_6_12', 'above_year', 'not_defined']
        entranceDate_order = list(reversed(entranceDate_order))
        condition_order = ['לא צויין', 'ישן', 'שמור', 'משופץ', 'חדש']

        # Initialize the OrdinalEncoder with the specified orders
        ordinal_encoder = OrdinalEncoder(categories=[type_order, furniture_order, entranceDate_order, condition_order])
        # Encode the specified columns
        columns_to_encode = ['type', 'furniture ', 'entranceDate ', 'condition ']
        data_encoded = ordinal_encoder.fit_transform(data[columns_to_encode])
        # Encode the categorical columns
        city_array = data['City_Combined'].values.reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(city_array).toarray()
        column_names = one_hot_encoder.get_feature_names_out(['City_Combined'])
        df_encoded = pd.DataFrame(one_hot_encoded, columns=column_names)
        # Convert data_encoded to a DataFrame with consistent column names
        data_encoded = pd.DataFrame(data_encoded, columns=columns_to_encode)

        data1.reset_index(inplace=True, drop=True)

        # Add the remaining columns
        data_final = pd.concat([data1, data_encoded, df_encoded], axis=1)
        # print(data_final)
        return data_final

    def impute_and_drop_columns(X):
        #     print(X.columns)
        imputer = KNNImputer(n_neighbors=5)
        columns_to_impute = ['room_number', 'Area', 'type']
        X_imputed = imputer.fit_transform(X[columns_to_impute])
        X_imputed = pd.DataFrame(X_imputed, columns=columns_to_impute)
        remaining_columns = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ','condition ','furniture ',
                             'hasBalcony ', 'hasMamad ', 'handicapFriendly ', 'entranceDate ', 'City_Combined_צפון',
                             'City_Combined_מרכז', 'City_Combined_דרום', 'City_Combined_תל אביב-זול',
                             'City_Combined_תל אביב-יקר', 'floor_ratio']
        #     X_remaining = X[remaining_columns]
        intersection_columns = list(set(X.columns).intersection(remaining_columns))
        X_remaining = X[intersection_columns]
        missing_columns = list(set(remaining_columns) - set(X.columns))
        X_missing = pd.DataFrame(0, index=np.arange(len(X)), columns=missing_columns)

        #     X_combined = pd.concat([X_imputed, X_remaining, X_missing], axis=1)
        #     X_remaining = X.loc[:, remaining_columns]
        X_transformed = pd.concat([X_imputed, X_remaining, X_missing], axis=1)
        #     print(X_transformed)
        return X_transformed

    def do_changes(data):

        data['City'] = data['City'].replace({"נהרייה": "נהריה", " נהרייה": "נהריה", " נהריה": "נהריה"})
        data['City_Combined'] = data.apply(combine_columns, axis=1)

        data['Area'] = data['Area'].apply(
            lambda x: int(re.findall('[0-9]+', str(x))[0]) if re.findall('[0-9]+', str(x)) else None)

        data['floor'] = data['floor_out_of'].apply(lambda x: to_floor(x))
        # ---------------------------------------------------------------------------------------------------------------------------
        data['total_floor'] = data['floor_out_of'].apply(lambda x: total_floor(x))

        # ---------------------------------------------------------------------------------------------------------------------------
        lst = ['hasElevator ',
               'hasParking ', 'hasBars ', 'hasStorage ',
               'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']

        for i in lst:
            data[i] = data[i].apply(lambda x: convert_to_binary(x))

            # ---------------------------------------------------------------------------------------------------------------------------
        data['room_number'] = data['room_number'].apply(lambda x: to_room_num(x))

        # ---------------------------------------------------------------------------------------------------------------------------
        data['type'] = data['type'].apply(lambda x: to_other_type(x))

        # ---------------------------------------------------------------------------------------------------------------------------
        data['entranceDate '] = data['entranceDate '].apply(lambda x: calculate_entrance_date(x))

        data['floor_ratio'] = data['floor'] / data['total_floor']
        data['floor_ratio'] = data['floor_ratio'].fillna(0)
        # ---------------------------------------------------------------------------------------------------------------------------
        data['condition '] = data['condition '].apply(lambda x: to_condition(x))

        data['furniture '] = data['furniture '].apply(lambda x: to_furniture(x))

        data['City_Combined'] = data['City_Combined'].apply(lambda x: get_region(str(x)))
        return data

    # מחיר קיצוני שמגדיל את הערכים של המחיר
    data = do_changes(data)
    data = encode_columns(data)
    data = impute_and_drop_columns(data)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled
