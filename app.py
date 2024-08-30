from flask import Flask, render_template, request
import plotly.graph_objs as go
import requests
import json
import wbpy
from sklearn.ensemble import RandomForestRegressor
app = Flask(__name__)

# Initialize the API
api = wbpy.IndicatorAPI()

iso_country_codes = {"Morocco": "MA", "France": "FR", "Japan": "JP" , "Algeria" : "DZ" , "Afghanistan": "AF",
    "Albania": "AL", "Andorra": "AD",
    "Angola": "AO",
    "Antigua and Barbuda": "AG",
    "Argentina": "AR",
    "Armenia": "AM",
    "Australia": "AU",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Bahamas": "BS",
    "Bahrain": "BH",
    "Bangladesh": "BD",
    "Barbados": "BB",
    "Belarus": "BY",
    "Belgium": "BE",
    "Belize": "BZ",
    "Benin": "BJ",
    "Bhutan": "BT",
    "Bolivia": "BO",
    "Bosnia and Herzegovina": "BA",
    "Botswana": "BW",
    "Brazil": "BR",
    "Brunei": "BN",
    "Bulgaria": "BG",
    "Burkina Faso": "BF",
    "Burundi": "BI",
    "Cabo Verde": "CV",
    "Cambodia": "KH",
    "Cameroon": "CM",
    "Canada": "CA",
    "Central African Republic": "CF",
    "Chad": "TD",
    "Chile": "CL",
    "China": "CN",
    "Colombia": "CO","Comoros": "KM",
    "Congo": "CG",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Cuba": "CU",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Djibouti": "DJ",
    "Dominica": "DM",
    "Dominican Republic": "DO",
    "Ecuador": "EC",
    "Egypt": "EG",
    "El Salvador": "SV",
    "Equatorial Guinea": "GQ",
    "Eritrea": "ER",
    "Estonia": "EE",
    "Eswatini": "SZ",
    "Ethiopia": "ET",
    "Fiji": "FJ",
    "Finland": "FI",
    "Gabon": "GA",
    "Gambia": "GM",
    "Georgia": "GE",
    "Germany": "DE",
    "Ghana": "GH",
    "Greece": "GR",
    "Grenada": "GD",
    "Guatemala": "GT",
    "Guinea": "GN",
    "Guinea-Bissau": "GW",
    "Guyana": "GY",
    "Haiti": "HT",
    "Honduras": "HN",
    "Hungary": "HU",
    "Iceland": "IS",
    "India": "IN",
    "Indonesia": "ID",
    "Iran": "IR",
    "Iraq": "IQ",
    "Ireland": "IE",
    "Israel": "IL",
    "Italy": "IT",
    "Jamaica": "JM",
    "Japan": "JP",
    "Jordan": "JO",
    "Kazakhstan": "KZ",
    "Kenya": "KE",
    "Kiribati": "KI",
    "Kuwait": "KW",
    "Kyrgyzstan": "KG",
    "Laos": "LA",
    "Latvia": "LV",
    "Lebanon": "LB",
    "Lesotho": "LS",
    "Liberia": "LR",
    "Libya": "LY",
    "Liechtenstein": "LI",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Madagascar": "MG",
    "Malawi": "MW",
    "Malaysia": "MY",
    "Maldives": "MV",
    "Mali": "ML",
    "Malta": "MT",
    "Marshall Islands": "MH",
    "Mauritania": "MR",
    "Mauritius": "MU",
    "Mexico": "MX",
    "Micronesia": "FM",
    "Moldova": "MD",
    "Monaco": "MC",
    "Mongolia": "MN",
    "Montenegro": "ME",
    "Mozambique": "MZ",
    "Myanmar": "MM",
    "Namibia": "NA",
    "Nauru": "NR",
    "Nepal": "NP",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Niger": "NE",
    "Nigeria": "NG",
    "North Korea": "KP",
    "North Macedonia": "MK",
    "Norway": "NO",
    "Oman": "OM",
    "Pakistan": "PK",
    "Palau": "PW",
    "Palestine" : "PA",
    "Panama": "PA",
    "Papua New Guinea": "PG",
    "Paraguay": "PY",
    "Peru": "PE",
    "Philippines": "PH",
    "Poland": "PL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Romania": "RO",
    "Russia": "RU",
    "Rwanda": "RW",
    "Saint Kitts and Nevis": "KN",
    "Saint Lucia": "LC",
    "Saint Vincent and the Grenadines": "VC",
    "Samoa": "WS",
    "San Marino": "SM",
    "Sao Tome and Principe": "ST",
    "Saudi Arabia": "SA",
    "Senegal": "SN",
    "Serbia": "RS",
    "Seychelles": "SC",
    "Sierra Leone": "SL",
    "Singapore": "SG",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Solomon Islands": "SB",
    "Somalia": "SO",
    "South Africa": "ZA",
    "South Korea": "KR",
    "South Sudan": "SS",
    "Spain": "ES",
    "Sri Lanka": "LK",
    "Sudan": "SD",
    "Suriname": "SR",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Syria": "SY",
    "Taiwan": "TW",
    "Tajikistan": "TJ",
    "Tanzania": "TZ",
    "Thailand": "TH",
    "Timor-Leste": "TL",
    "Togo": "TG",
    "Tonga": "TO",
    "Trinidad and Tobago": "TT",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Turkmenistan": "TM",
    "Tuvalu": "TV",
    "Uganda": "UG",
    "Ukraine": "UA",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United States": "US",
    "Uruguay": "UY",
 
    }
total_population = "SP.POP.TOTL"
total_CO2 = "EN.ATM.CO2E.KT"
air_quality= "EN.ATM.PM25.MC.M3"
water = "ER.GDP.FWTL.M3.KD"
inflation = "FP.CPI.TOTL.ZG"
unemployment ="SL.UEM.TOTL.ZS"
energy ="NY.GDP.PCAP.CD"
employment="SL.EMP.MPYR.ZS"
female="SL.UEM.TOTL.MA.ZS"
male="SL.UEM.TOTL.FE.ZS"
m_agri="SL.AGR.EMPL.MA.ZS"
m_indus="SL.IND.EMPL.MA.ZS"
m_serv="SL.SRV.EMPL.MA.ZS"
f_agri="SL.AGR.EMPL.FE.ZS"
f_serv="SL.SRV.EMPL.FE.ZS"
f_indus="SL.IND.EMPL.FE.ZS"
internet="IT.CEL.SETS.P2"
air="IS.AIR.GOOD.MT.K1"
rail="IS.RRS.GOOD.MT.K6"
air_pass="IS.AIR.PSGR"
rail_pass="IS.RRS.PASG.KM"
mortality="SH.STA.MMRT"
life_exp_tot="SP.DYN.LE00.IN"
birth_rate="SP.DYN.CBRT.IN"
death_rate="SP.DYN.CDRT.IN"
fertility_rate="SP.DYN.TFRT.IN"
life_exp_male="SP.DYN.LE00.FE.IN"
life_exp_female="SP.DYN.LE00.MA.IN"

def get_country_infos(api , country_name):
    headers = {'X-Api-Key': '4MRMJM/1GfJtwzfyawzpEg==gkaVEQtaIFJslF9H'}
    response = requests.get(api, headers=headers)
    if response.status_code == requests.codes.ok:
        return response.json()
    else:
        print("No data found")

def get_country_populations(country_codes):
    dataset = api.get_dataset(total_population, iso_country_codes.values(), date="2010:2020")
    country_pop = dataset.as_dict()
    for country in country_pop.keys():
        if country == country_codes:
            return country_pop[country]

def get_predicted_populations():
    dataset = api.get_dataset(total_population, iso_country_codes.values(), date="2010:2020")
    country_pop = dataset.as_dict()

        
def get_country_co2(country_codes):
    dataset = api.get_dataset(total_CO2, iso_country_codes.values(), date="2010:2020")
    country_emission = dataset.as_dict()
    for country in country_emission.keys():
        if country == country_codes:
            return country_emission[country]

def get_country_air(country_codes):
    dataset = api.get_dataset(air_quality, iso_country_codes.values(), date="2010:2019")
    country_air = dataset.as_dict()
    for country in country_air.keys():
        if country == country_codes:
            return country_air[country]

def get_country_water(country_codes):
    dataset_water = api.get_dataset(water, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_internet(country_codes):
    dataset_water = api.get_dataset(internet, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]



def get_country_air_pass(country_codes):
    dataset_water = api.get_dataset(air_pass, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]



def get_country_air_goods(country_codes):
    dataset_water = api.get_dataset(air, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_rail_goods(country_codes):
    dataset_water = api.get_dataset(rail, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_rail_pass(country_codes):
    dataset_water = api.get_dataset(rail_pass, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_n2o(country_codes):
    dataset = api.get_dataset("EN.ATM.NOXE.KT.CE", iso_country_codes.values(), date="2010:2020")
    country_n2o = dataset.as_dict()
    for country in country_n2o.keys():
        if country == country_codes:
            return country_n2o[country]

def get_country_methane(country_codes):
    dataset = api.get_dataset("EN.ATM.METH.KT.CE", iso_country_codes.values(), date="2010:2020")
    country_methane = dataset.as_dict()
    for country in country_methane.keys():
        if country == country_codes:
            return country_methane[country]

def get_country_infla(country_codes):
    dataset_water = api.get_dataset(inflation, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_unemployment(country_codes):
    dataset_water = api.get_dataset(unemployment, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_employment(country_codes):
    dataset_water = api.get_dataset(employment, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_energy(country_codes):
    dataset_water = api.get_dataset(energy, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

#######################

def get_country_female(country_codes):
    dataset_water = api.get_dataset(female, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_male(country_codes):
    dataset_water = api.get_dataset(male, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]
def get_country_fagr(country_codes):
    dataset_water = api.get_dataset(f_agri, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_findus(country_codes):
    dataset_water = api.get_dataset(f_indus, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_fserv(country_codes):
    dataset_water = api.get_dataset(f_serv, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]
def get_country_fagr(country_codes):
    dataset_water = api.get_dataset(f_agri, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_findus(country_codes):
    dataset_water = api.get_dataset(f_indus, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_fserv(country_codes):
    dataset_water = api.get_dataset(f_serv, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_mindus(country_codes):
    dataset_water = api.get_dataset(m_indus, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_mserv(country_codes):
    dataset_water = api.get_dataset(m_serv, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_magri(country_codes):
    dataset_water = api.get_dataset(m_agri, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]
#################################
def get_country_life(country_codes):
    dataset_water = api.get_dataset(life_exp_tot, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_life_male(country_codes):
    dataset_water = api.get_dataset(life_exp_male, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_life_female(country_codes):
    dataset_water = api.get_dataset(life_exp_female, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_birth(country_codes):
    dataset_water = api.get_dataset(birth_rate, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_death(country_codes):
    dataset_water = api.get_dataset(death_rate, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]


def get_country_fertility(country_codes):
    dataset_water = api.get_dataset(fertility_rate, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

def get_country_mortality(country_codes):
    dataset_water = api.get_dataset(mortality, iso_country_codes.values(), date="2010:2020")
    country_water = dataset_water.as_dict()
    for country in country_water.keys():
        if country == country_codes:
            return country_water[country]

from tensorflow.keras.models import Sequential
from sklearn.linear_model import LinearRegression
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
def prepare_data(population_data):
    years = []
    populations = []
    for year, population in population_data.items():
        if population is not None:
            years.append(int(year))
            populations.append(population)
    X = np.array(years).reshape(-1, 1)
    y = np.array(populations)
    return X, y

def train_model_pop(X_train, y_train):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])) # Ajouter une dimension pour correspondre aux exigences de l'entrée LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model
    
from sklearn.svm import SVR
def train_model_random(X_train, y_train):
    model = SVR(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    return model

def predict_populations(model, years):
    X_pred = np.array(years).reshape(-1, 1)
    population_pred = model.predict(X_pred)
    return population_pred

@app.route('/base', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        country_name = request.form['country_name']
        country_code = iso_country_codes.get(country_name)
        country_pic = f"https://flagsapi.com/{country_code}/flat/64.png"
        api_url = f"https://api.api-ninjas.com/v1/country?name={country_name}"
        country_infos = get_country_infos(api_url , country_code)
        if country_code:
    # Get country data
            country_data = get_country_populations(country_code)
            country_data = dict(reversed(country_data.items()))
            country_emission = get_country_co2(country_code)
            country_emission = dict(reversed(country_emission.items()))
            country_air = get_country_air(country_code)
            country_air = dict(reversed(country_air.items()))
            country_water = get_country_water(country_code)
            country_water = dict(reversed(country_water.items()))
    
    # Prepare data for modeling
            X, y = prepare_data(country_data)
            X_train, y_train = X, y  # Assuming we're using all data for training
    
    # Train model
            models = train_model_pop(X_train, y_train)
    
    # Define years for prediction
            years_to_predict = [2021, 2022, 2023, 2024, 2025]
    
    # Make predictions
            predictions = predict_populations(models, years_to_predict)

    # Prepare figures
            population_fig = go.Bar(x=list(country_data.keys()), y=list(country_data.values()), name='Population Data', marker=dict(color='rgba(55, 128, 191, 0.7)'))
            emission_fig = go.Bar(x=list(country_emission.keys()), y=list(country_emission.values()), name='CO2 Emission Data', marker=dict(color='rgba(255, 153, 51, 0.7)'))
            air_fig = go.Bar(x=list(country_emission.keys()), y=list(country_air.values()), name='air quality Data', marker=dict(color='rgba(255, 153, 51, 0.7)'))
            water_fig = water_fig = go.Scatter(x=list(country_water.keys()), y=list(country_water.values()), name='Water Data', mode='lines+markers', marker=dict(color='rgba(255, 153, 51, 0.7)'))

            predictions_fig = go.Bar(x=years_to_predict, y=predictions, name='Predictions', marker=dict(color='red'))

    # Define layout for population figure
            layout_pop = go.Layout(title='Population Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Population'))
    
    # Create population figure
            pop_fig = go.Figure(data=[population_fig, predictions_fig], layout=layout_pop)
        

    # Define layout for emission figure
            layout_emi = go.Layout(title='Emission Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Emission'))
    
    # Create emission figure
            emi_fig = go.Figure(data=[emission_fig], layout=layout_emi)

    #
            layout_air = go.Layout(title='air quality Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='air quality'))

    #
            air_fig = go.Figure(data=[air_fig], layout=layout_air)

    #
            layout_water = go.Layout(title='water Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='water'))

            water_fig = go.Figure(data=[water_fig], layout=layout_water)

    #



    # Convert figures to JSON
            graph_pop = pop_fig.to_json()
            graph_emi = emi_fig.to_json()
            graph_air = air_fig.to_json()
            graph_water = water_fig.to_json()
            #graph_json_str = json.dumps(graph_json)
            country_info = country_infos[0]
            gdp = country_info.get("gdp", "GDP not found")
            capital = country_info.get("capital", "Capital not found")
            population = country_info.get("population", "Population not found")
            currency_name = country_info.get("currency", {}).get("name", "Currency name not found")

            return render_template('base.html', gdp=gdp, capital=capital, population=population, currency_name=currency_name,country_informations=country_infos , country_pic= country_pic , country_name=country_name, country_data=country_data, country_emission=country_emission , graph_json_pop=graph_pop , graph_json_emi=graph_emi,predictions=predictions , graph_json_water=graph_water , graph_json_air=graph_air )
        else:
            return "Country not found!"
    return render_template('index.html')

@app.route('/climatique', methods=['GET', 'POST'])
def climatique():
        country_name = request.args.get('country_name')
        country_code = iso_country_codes.get(country_name)
        country_pic = f"https://flagsapi.com/{country_code}/flat/64.png"
        api_url = f"https://api.api-ninjas.com/v1/country?name={country_name}"
        country_infos = get_country_infos(api_url, country_code)
        
        if country_code:
            # Get country data
            country_data = get_country_populations(country_code)
            country_data = dict(reversed(country_data.items()))
            country_emission = get_country_co2(country_code)
            country_emission = dict(reversed(country_emission.items()))
            country_air = get_country_air(country_code)
            country_air = dict(reversed(country_air.items()))
            country_water = get_country_water(country_code)
            country_water = dict(reversed(country_water.items()))
            country_n2o=get_country_n2o(country_code)
            country_n2o = dict(reversed(country_n2o.items()))
            country_methane = get_country_methane(country_code)
            country_methane = dict(reversed(country_methane.items()))
        
            # Prepare data for modeling
            X, y = prepare_data(country_data)
            X_train, y_train = X, y  # Assuming we're using all data for training
            
            # Train model
            model = train_model_pop(X_train, y_train)
            
            # Define years for prediction
            years_to_predict = [2021, 2022, 2023, 2024, 2025]
            
            # Make predictions
            predictions = predict_populations(model, years_to_predict)
            #################
            X, y = prepare_data(country_emission)
            X_train, y_train = X, y  # Assuming we're using all data for training
            
            # Train model
            model = train_model_pop(X_train, y_train)
            
            # Define years for prediction
            years_to_predict = [2021, 2022, 2023, 2024, 2025]
            
            # Make predictions
            predictions_emission = predict_populations(model, years_to_predict)
            ##################
            X, y = prepare_data(country_air)
            X_train, y_train = X, y  # Assuming we're using all data for training
            
            # Train model
            model = train_model_random(X_train, y_train)
            
            # Define years for prediction
            years_to_predict = [2021, 2022, 2023, 2024, 2025]
            year_to_predict = [2020,2021,2022,2023,2024,2025]
            # Make predictions
            predictions_air = predict_populations(model, years_to_predict)
            # Prepare figures
            population_fig = go.Bar(x=list(country_data.keys()), y=list(country_data.values()), name='Population Data', marker=dict(color='rgba(55, 128, 191, 0.7)'))
            emission_fig = go.Bar(x=list(country_emission.keys()), y=list(country_emission.values()), name='CO2 Emission Data', marker=dict(color='rgba(255, 153, 51, 0.7)'))
            air_fig = go.Bar(x=list(country_air.keys()), y=list(country_air.values()), name='Air Quality Data', marker=dict(color='rgba(255, 153, 51, 0.7)'))
            water_fig = go.Scatter(x=list(country_water.keys()), y=list(country_water.values()), name='Water Data', mode='lines+markers', marker=dict(color='rgba(255, 153, 51, 0.7)'))
            predictions_fig = go.Bar( visible=False,x=years_to_predict, y=predictions, name='Predictions', marker=dict(color='red'))
            prediction_air_fig= go.Bar(x=year_to_predict, y=predictions_air, name='Predictions', marker=dict(color='red'))
            predictions_emission_fig = go.Bar( visible=False,x=years_to_predict, y=predictions_emission , name='Predictions', marker=dict(color='red'))
            X, y = prepare_data(country_water)

            X_train, y_train = X, y  # Assuming we're using all data for training
            
            # Train model
            model = train_model_random(X_train, y_train)
            
            # Define years for prediction
            years_to_predict = [2021, 2022, 2023, 2024, 2025]
            
            # Make predictions
            predictions_water = predict_populations(model, years_to_predict)
            prediction_water_fig =  go.Bar( visible=False,x=years_to_predict, y=predictions_water , name='Predictions', marker=dict(color='red'))
            # Define layout for population figure
            
            
            layout_pop = go.Layout(title='Population Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Population'))
            
            # Create population figure
            pop_fig = go.Figure(data=[population_fig, predictions_fig], layout=layout_pop)
            
            # Define layout for emission figure
            layout_emi = go.Layout(title='CO2 Emission', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Emission'))
            
            # Create emission figure
            emi_fig = go.Figure(data=[emission_fig,predictions_emission_fig], layout=layout_emi)
            
            # Define layout for air quality figure
            layout_air = go.Layout(title='Air Quality Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Air Quality'))
            
            # Create air quality figure
            air_fig = go.Figure(data=[air_fig,prediction_air_fig], layout=layout_air)
            
            # Define layout for water figure
            layout_water = go.Layout(title='Water Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Water'))
            
            # Create water figure
            water_fig = go.Figure(data=[water_fig,prediction_water_fig], layout=layout_water)
            
            # Convert figures to JSON
            graph_pop = pop_fig.to_json()
            graph_emi = emi_fig.to_json()
            graph_air = air_fig.to_json()
            graph_water = water_fig.to_json()
            country_info = country_infos[0]
            co2 = country_info.get("co2_emissions", "CO2 emissions not found")
            species = country_info.get("threatened_species", "Threatened species not found")
            area = country_info.get("forested_area", "Forested area not found")
            labels = ['CO2', 'N2O', 'Methane']
            values = [sum(country_emission.values()), sum(country_n2o.values()),sum(country_methane.values())]

            # Create pie chart
            pie_chart = go.Pie(labels=labels, values=values, name='Gaz Emission', marker=dict(colors=['rgba(255, 99, 71, 0.7)', 'rgba(60, 179, 113, 0.7)','rgba(255, 153, 51, 0.7)']))

            # Define layout for the pie chart
            layout_pie = go.Layout(title='Gaz Emission')

            # Create pie chart figure
            pie_fig = go.Figure(data=[pie_chart], layout=layout_pie)

            # Convert the figure to JSON if needed
            graph_pie = pie_fig.to_json()
            return render_template('climatique.html', graph_pie=graph_pie,predictions_water=predictions_water,prediction_air_fig=prediction_air_fig,predictions_emission=predictions_emission,country_informations=country_infos, country_pic=country_pic, country_name=country_name, country_data=country_data, country_emission=country_emission, graph_json_pop=graph_pop, graph_json_emi=graph_emi, predictions=predictions, graph_json_water=graph_water, graph_json_air=graph_air , co2=co2 , species=species , area=area)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')
@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')
@app.route('/news', methods=['GET', 'POST'])
def news():
    return render_template('news.html')
@app.route('/infra', methods=['GET', 'POST'])
def infra():
    country_name = request.args.get('country_name')
    country_code = iso_country_codes.get(country_name)
    country_pic = f"https://flagsapi.com/{country_code}/flat/64.png"
    api_url = f"https://api.api-ninjas.com/v1/country?name={country_name}"
    country_infos = get_country_infos(api_url, country_code)
    
    
    if country_code:
        # Get country data
        country_internet = get_country_internet(country_code)
        country_internet = dict(reversed(country_internet.items()))
        
        # Plotly graph
        country_internet_fig = go.Bar(
            x=list(country_internet.keys()), 
            y=list(country_internet.values()), 
            name='Internet Data', 
            marker=dict(color='rgba(55, 128, 191, 0.7)')
        )
        layout_internet = go.Layout(
            title='Internet Servers Data', 
            barmode='group', 
            xaxis=dict(title='Year'), 
            yaxis=dict(title='Million')
        )
        X, y = prepare_data(country_internet)
        X_train, y_train = X, y  # Assuming we're using all data for training # Train model
        model = train_model_random(X_train, y_train)# Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
        predictions_internet = predict_populations(model, years_to_predict)
        prediction_internet_fig = go.Bar(x=list(years_to_predict), y=list(predictions_internet), name='Prediction', marker=dict(color='red'))
        
        # Create figure
        internet_fig = go.Figure(data=[country_internet_fig,prediction_internet_fig], layout=layout_internet)
        graph_internet = internet_fig.to_json()
        #####################################
        country_rail = get_country_rail_goods(country_code)
        country_rail = dict(reversed(country_rail.items()))
        rail_fig = go.Scatter(x=list(country_rail.keys()), y=list(country_rail.values()), name='Country Rail Data', mode='lines+markers', marker=dict(color='rgba(255, 153, 51, 0.7)'))
        layout_rail = go.Layout(title='Country Rail Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='KM'))
        X, y = prepare_data(country_rail)
        X_train, y_train = X, y  # Assuming we're using all data for training # Train model
        model = train_model_random(X_train, y_train)# Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
        predictions_rail = predict_populations(model, years_to_predict)
        prediction_rail_fig = go.Scatter(x=list(years_to_predict), y=list(predictions_rail), name='Prediction', mode='lines+markers', marker=dict(color='red'))
        country_rail_fig = go.Figure(data=[rail_fig,prediction_rail_fig], layout=layout_rail)
        graph_rail = country_rail_fig.to_json()
        #####################################
        country_air = get_country_air_goods(country_code)
        country_air = dict(reversed(country_air.items()))
        trans_air_fig = go.Scatter(x=list(country_air.keys()), y=list(country_air.values()), name='Air Transport (Million)', mode='lines', marker=dict(color='lightblue'))
        layout_trans_air = go.Layout(title='Air Transport Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Million'))
        X, y = prepare_data(country_air)
        X_train, y_train = X, y  # Assuming we're using all data for training # Train model
        model = train_model_random(X_train, y_train)# Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
        predictions_air = predict_populations(model, years_to_predict)
        prediction_air_fig = go.Scatter(x=list(years_to_predict), y=list(predictions_air), name='Prediction', mode='lines+markers', marker=dict(color='red'))
        trans_air_fig = go.Figure(data=[trans_air_fig,prediction_air_fig], layout=layout_trans_air)
        trans_graph_air = trans_air_fig.to_json()
        #####################################
        
    return render_template('infra.html', predictions_internet=predictions_internet,
                           graph_internet=graph_internet,
                           country_informations=country_infos, 
                           country_pic=country_pic, 
                           country_name=country_name,
                           graph_rail=graph_rail,
                           trans_graph_air=trans_graph_air)


@app.route('/health', methods=['GET', 'POST'])
def health():
    country_name = request.args.get('country_name')
    country_code = iso_country_codes.get(country_name)
    country_pic = f"https://flagsapi.com/{country_code}/flat/64.png"
    api_url = f"https://api.api-ninjas.com/v1/country?name={country_name}"
    country_infos = get_country_infos(api_url, country_code)
    if country_code:
        ########################
        country_fer = get_country_fertility(country_code)
        country_fer = dict(reversed(country_fer.items()))
        fer_fig =go.Scatter(x=list(country_fer.keys()), y=list(country_fer.values()), name='Country Fertility Data', mode='lines+markers', marker=dict(color='rgba(255, 153, 51, 0.7)'))
        layout_fer=go.Layout(title='Fertility Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Birth per women'))
        X, y = prepare_data(country_fer)
        X_train, y_train = X, y  # Assuming we're using all data for training # Train model
        model = train_model_random(X_train, y_train)# Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
        predictions_fer = predict_populations(model, years_to_predict)
        prediction_fer_fig = go.Scatter(x=list(years_to_predict), y=list(predictions_fer), name='Prediction', mode='lines+markers', marker=dict(color='red'))
        country_fer_fig= go.Figure(data=[fer_fig,prediction_fer_fig], layout=layout_fer)
        graph_fer=country_fer_fig.to_json()
        #########################
        country_life = get_country_life(country_code)
        country_life = dict(reversed(country_life.items()))
        life_fig = go.Scatter(x=list(country_life.keys()), y=list(country_life.values()), name='Country life Data', mode='lines+markers', marker=dict(color='rgba(255, 153, 51, 0.7)'))
        layout_life = go.Layout(title='Country life expectancy Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Years'))
        X, y = prepare_data(country_life)
        X_train, y_train = X, y  # Assuming we're using all data for training # Train model
        model = train_model_random(X_train, y_train)# Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
        predictions_life = predict_populations(model, years_to_predict)
        prediction_life_fig = go.Scatter(x=list(years_to_predict), y=list(predictions_life), name='Prediction', mode='lines+markers', marker=dict(color='red'))
        country_life_fig = go.Figure(data=[life_fig,prediction_life_fig], layout=layout_life)
        graph_life = country_life_fig.to_json()

        #########################
        country_death = get_country_death(country_code)
        country_death = dict(reversed(country_death.items()))
        death_fig = go.Scatter(x=list(country_death.keys()), y=list(country_death.values()), name='Country life Data', mode='lines+markers', marker=dict(color='lightgreen'))
        layout_death = go.Layout(title='Country Death expectancy Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Years'))
        X, y = prepare_data(country_death)
        X_train, y_train = X, y  # Assuming we're using all data for training # Train model
        model = train_model_random(X_train, y_train)# Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
        predictions_death = predict_populations(model, years_to_predict)
        prediction_death_fig = go.Scatter(x=list(years_to_predict), y=list(predictions_death), name='Prediction', mode='lines+markers', marker=dict(color='red'))
        country_death_fig = go.Figure(data=[death_fig,prediction_death_fig], layout=layout_death)
        graph_death = country_death_fig.to_json()
        #########################   
        country_life_male = get_country_life_male(country_code)
        country_life_female = get_country_life_female(country_code)
        country_life_male = dict(reversed(country_life_male.items()))
        country_life_female = dict(reversed(country_life_female.items()))
# Prepare data for the pie chart
        labels = ['Male', 'Female']
        values = [sum(country_life_male.values()), sum(country_life_female.values())]

# Create the pie chart
        pie_fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=['blue', 'pink']))])

# Set the layout for the pie chart
        pie_fig.update_layout(title='Country Life Expectancy: Male vs Female')

# Convert the figure to JSON
        graph_pie = pie_fig.to_json()

    return render_template('health.html',graph_pie=graph_pie,graph_death=graph_death,graph_life=graph_life,graph_fer=graph_fer ,country_informations=country_infos, country_pic=country_pic, country_name=country_name)







@app.route('/socioeconomic', methods=['GET', 'POST'])
def socioeconomic():
    country_name = request.args.get('country_name')
    country_code = iso_country_codes.get(country_name)
    country_pic = f"https://flagsapi.com/{country_code}/flat/64.png"
    api_url = f"https://api.api-ninjas.com/v1/country?name={country_name}"
    country_infos = get_country_infos(api_url, country_code)
    country_name = request.args.get('country_name')
    if country_code:
            # Get country data
        country_data = get_country_energy(country_code)
        country_data = dict(reversed(country_data.items()))
        country_emission = get_country_infla(country_code)
        country_emission = dict(reversed(country_emission.items()))
        country_air = get_country_unemployment(country_code)
        country_air = dict(reversed(country_air.items()))
        country_emp = get_country_employment(country_code)
        country_emp = dict(reversed(country_emp.items()))
        country_fem= get_country_female(country_code)
        country_fem = dict(reversed(country_fem.items()))
        country_male=get_country_male(country_code)
        country_male = dict(reversed(country_male.items()))
            ##########
        m_indus=get_country_mindus(country_code)
        m_indus=dict(reversed(m_indus.items()))
        m_serv=get_country_mserv(country_code)
        m_serv=dict(reversed(m_indus.items())) 
        m_agri=get_country_magri(country_code)
        m_agri=dict(reversed(m_indus.items()))
            ########
        f_indus=get_country_findus(country_code)
        f_indus=dict(reversed(f_indus.items()))
        f_serv=get_country_fserv(country_code)
        f_serv=dict(reversed(f_serv.items())) 
        f_agri=get_country_fagr(country_code)
        f_agri=dict(reversed(f_agri.items()))   





            
            # Prepare data for modeling
        X, y = prepare_data(country_data)
        X_train, y_train = X, y  # Assuming we're using all data for training
            
            # Train model
        model = train_model_pop(X_train, y_train)
            
            # Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
            
            # Make predictions
        predictions = predict_populations(model, years_to_predict)
            
            # Prepare figures
        population_fig = go.Bar(x=list(country_data.keys()), y=list(country_data.values()), name='GDP', marker=dict(color='rgba(55, 128, 191, 0.7)'))
        prediction_population_fig = go.Bar(x=list(years_to_predict), y=list(predictions), name='Prediction', marker=dict(color='red'))

        emission_fig = go.Scatter(x=list(country_emission.keys()), y=list(country_emission.values()), name='Inflation', marker=dict(color='rgba(255, 153, 51, 0.7)'))
        X, y = prepare_data(country_emission)
        X_train, y_train = X, y  # Assuming we're using all data for training
            
            # Train model
        model = train_model_pop(X_train, y_train)
            
            # Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
            
            # Make predictions
        predictions_emission = predict_populations(model, years_to_predict)
        prediction_emission_fig = go.Scatter(x=list(years_to_predict), y=list(predictions_emission), name='Predictions', marker=dict(color='red'))

        air_fig = go.Bar(x=list(country_air.keys()), y=list(country_air.values()), name='Unemployed Data', marker=dict(color='rgba(255, 153, 51, 0.7)'))
        X, y = prepare_data(country_emission)
        X_train, y_train = X, y  # Assuming we're using all data for training
            
            # Train model
        model = train_model_random(X_train, y_train)
            
            # Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
            
            # Make predictions
        predictions_unemployement = predict_populations(model, years_to_predict)
        air_prediction_fig = go.Bar(x=list(years_to_predict), y=list(predictions_unemployement), name='prediction', marker=dict(color='red'))

        air_emp = go.Bar(x=list(country_emp.keys()), y=list(country_emp.values()), name='Employement Data', marker=dict(color='rgba(255, 153, 51, 0.7)'))
            
            # Define layout for population figure
        layout_pop = go.Layout(title='GDP Per Capita', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='USD'))
            # Create population figure
        pop_fig = go.Figure(data=[population_fig,prediction_population_fig], layout=layout_pop)
            
            # Define layout for emission figure
        layout_emi = go.Layout(title='Inflation Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='USD'))
            
            # Create emission figure
        emi_fig = go.Figure(data=[emission_fig,prediction_emission_fig], layout=layout_emi)
            
            # Define layout for air quality figure
        layout_air = go.Layout(title='Unemployment Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='% Citizens'))
            
            # Create air quality figure
        air_fig = go.Figure(data=[air_fig,air_prediction_fig], layout=layout_air)

            ##########

        layout_emp = go.Layout(title='Employment Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='% Citizens'))
            
            # Create air quality figure
        air_emp = go.Figure(data=[air_emp], layout=layout_emp)
        labels = ['Unemployed', 'Employed']


        values = [sum(country_emp.values()), sum(country_air.values())]

            # Create pie chart
        pie_chart = go.Pie(labels=labels, values=values, name='Employment Data', marker=dict(colors=['rgba(255, 99, 71, 0.7)', 'rgba(60, 179, 113, 0.7)']))

            # Define layout for the pie chart
        layout_pie = go.Layout(title='Employment vs Unemployment Data')

            # Create pie chart figure
        pie_fig = go.Figure(data=[pie_chart], layout=layout_pie)

            # Convert the figure to JSON if needed
        graph_pie = pie_fig.to_json()
            

            ############

        country_male = go.Bar(
            x=list(country_male.keys()), 
            y=list(country_male.values()), 
            name='Unemployed Male', 
            marker=dict(color='rgba(255, 153, 51, 0.7)')
        )

        country_fem = go.Bar(
            x=list(country_fem.keys()), 
            y=list(country_fem.values()), 
            name='Unemployed Male', 
            marker=dict(color='rgba(0, 102, 204, 0.7)')  # Different color for distinction
        )

            # Create the figure and add the bar charts
        fig_gender = go.Figure(data=[country_male, country_fem])

            # Update the layout for grouped bars
        fig_gender.update_layout(
            barmode='group',
            title='Male vs Female Data',
            xaxis_title='Year',
            yaxis_title='%',
            legend_title='Data Type'
        )

            # Show the figure

        combined_data_female = {
        'Industry': sum(f_indus.values()),
        'Services': sum(f_serv.values()),
        'Agriculture': sum(f_agri.values())  }

            # Create the pie chart
        fig_female = go.Figure(data=[go.Pie(
            labels=list(combined_data_female.keys()),
            values=list(combined_data_female.values()),
            hole=0.3  # This makes it a donut chart; remove for a regular pie chart
        )])

            # Update the layout for the chart
        fig_female.update_layout(
            title='Distribution of Industry, Services, and Agriculture',
        )


            #######################################"
        combined_data_male = {
        'Industry': sum(m_indus.values()),
        'Services': sum(m_serv.values()),
        'Agriculture': sum(m_agri.values())  }

            # Create the pie chart
        fig_male = go.Figure(data=[go.Pie(
            labels=list(combined_data_male.keys()),
            values=list(combined_data_male.values()),
            hole=0.3  # This makes it a donut chart; remove for a regular pie chart
        )])

            # Update the layout for the chart
        fig_male.update_layout(
            title='Distribution of Industry, Services, and Agriculture',
        )


        population_data = get_country_populations(country_code)
        population_data = dict(sorted(country_data.items()))
            
    
    # Prepare data for modeling
        X, y = prepare_data(population_data)
        X_train, y_train = X, y  # Assuming we're using all data for training
    
    # Train model
        models = train_model_pop(X_train, y_train)
    
    # Define years for prediction
        years_to_predict = [2021, 2022, 2023, 2024, 2025]
    
    # Make predictions
        predictions = predict_populations(models, years_to_predict)

    # Prepare figures
        population_fig = go.Bar(x=list(population_data.keys()), y=list(population_data.values()), name='Population Data', marker=dict(color='rgb(54,138,99)'))
        
        predictions_fig = go.Bar(x=years_to_predict, y=predictions, name='Predictions', marker=dict(color='red'))

    # Define layout for population figure
        layout_population = go.Layout(title='Population Data', barmode='group', xaxis=dict(title='Year'), yaxis=dict(title='Population'))
    
    # Create population figure
        population_fig = go.Figure(data=[population_fig, predictions_fig], layout=layout_population)
        

    # Define layout for emission figure
        



    # Convert figures to JSON
        graph_population = population_fig.to_json()



            # Define layout for water figure
            
            # Create water figure
            
            # Convert figures to JSON
        graph_pop = pop_fig.to_json()
        graph_emi = emi_fig.to_json()
        graph_air = air_fig.to_json()
        graph_emp = air_emp.to_json()
        graph_gender= fig_gender.to_json()
        graph_female= fig_female.to_json()
        graph_male=  fig_male.to_json()



        country_info = country_infos[0]
        gdp = country_info.get("gdp", "GDP not found")
        employment_industry = country_info.get("employment_industry", "Employment in industry not found")
        surface_area = country_info.get("surface_area", "Surface area not found")
        print(graph_population)
        return render_template('socioeconomic.html', graph_population=graph_population,air_prediction_fig=air_prediction_fig,predictions_emission=predictions_emission,country_informations=country_infos, country_pic=country_pic, country_name=country_name, country_data=country_data, country_emission=country_emission, graph_json_pop=graph_pop, graph_json_emi=graph_emi, predictions=predictions, graph_json_air=graph_air , gdp=gdp , employment_industry=employment_industry , surface_area=surface_area , graph_emp=graph_emp ,graph_pie=graph_pie , graph_gender=graph_gender , graph_female=graph_female , graph_male=graph_male)
    














if __name__ == '__main__':
    app.run(debug=True)