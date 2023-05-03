import streamlit as st
import pandas as pd
import numpy as np
import joblib
#import pickle

from sklearn.ensemble import ExtraTreesClassifier

def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)

model = joblib.load(r'../models/RTA_model.joblib')
#model = pickle.load(open('../models/RTA_model.pkl', 'rb'))

st.set_page_config(page_title="Accident Severity Prediction Module",
                   page_icon="🚧", layout="wide")

##Note for next project
# during the EDA I'm going to create a json file with this options, in order to create these lists
#automatically, not hardcoding.

#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']

options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']

options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']

options_vehicle_movement = ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover',
       'Waiting to go', 'Getting off', 'Reversing', 'Unknown', 'Parked',
       'Stopping', 'Overtaking', 'Other', 'Entering a junction']

options_road_alligment = ['Tangent road with flat terrain', 'Unknown',
       'Tangent road with mild grade and flat terrain', 'Escarpments',
       'Tangent road with rolling terrain', 'Gentle horizontal curve',
       'Tangent road with mountainous terrain and',
       'Steep grade downward with mountainous terrain',
       'Sharp reverse curve',
       'Steep grade upward with mountainous terrain']

options_type_of_collision = ['Collision with roadside-parked vehicles',
       'Vehicle with vehicle collision',
       'Collision with roadside objects', 'Collision with animals',
       'Other', 'Rollover', 'Fall from vehicles',
       'Collision with pedestrians', 'With Train', 'Unknown']

features=['number_of_vehicles_involved', 'number_of_casualties',
       'accident_severity', 'time_hours', 'day_of_week_encoded',
       'driving_experience_encoded', 'educational_level_encoded',
       'age_band_of_casualty_encoded', 'service_year_of_vehicle_encoded',
       'age_band_of_driver_encoded', 'casualty_severity_encoded',
       'cause_of_accident_encoded', 'type_of_vehicle_encoded',
       'area_accident_occured_encoded', 'vehicle_movement_encoded',
       'road_allignment_encoded', 'type_of_collision_encoded',
       'pedestrian_movement_encoded', 'weather_conditions_encoded',
       'types_of_junction_encoded', 'lanes_or_medians_encoded',
       'road_surface_type_encoded', 'owner_of_vehicle_encoded',
       'road_surface_conditions_encoded', 'casualty_class_encoded',
       'light_conditions_encoded', 'sex_of_casualty_encoded',
       'sex_of_driver_encoded']

lstAccidentSeverity = ['Slight Injury','Serious Injury','Fatal Injury']

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App</h1>", unsafe_allow_html=True)
st.write('<hr>', unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):

        st.subheader("Pick an input value for following features:")
        st.write('<hr>', unsafe_allow_html=True)
        
        hour = st.slider("Select Hour: ", 0, 23, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        casualties = st.slider("Select Casualities: ", 1, 8, value=0, format="%d")
        accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        vehicles_involved = st.slider("Select Vehicles involved: ", 1, 7, value=0, format="%d")
        vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        vehicle_movement =  st.selectbox("Select Vehicle movement: ", options=options_vehicle_movement)
        road_alligment =  st.selectbox("Select road alligment: ", options=options_road_alligment)
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
        accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
        driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        type_of_collision = st.selectbox("Select type of collision: ", options=options_type_of_collision)
        
        st.write('<hr>', unsafe_allow_html=True)
        submit = st.form_submit_button("Predict")


    if submit:
        day_of_week = ordinal_encoder(day_of_week, options_day)
        accident_cause = ordinal_encoder(accident_cause, options_cause)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        driver_age =  ordinal_encoder(driver_age, options_age)
        accident_area =  ordinal_encoder(accident_area, options_acc_area)
        driving_experience = ordinal_encoder(driving_experience, options_driver_exp) 
        lanes = ordinal_encoder(lanes, options_lanes)
        vehicle_movement = ordinal_encoder(vehicle_movement, options_vehicle_movement)
        road_alligment = ordinal_encoder(road_alligment, options_road_alligment)
        type_of_collision = ordinal_encoder(type_of_collision, options_type_of_collision)
                
        #there are 27 variables, I mapped just 13 of them in order to show the process.
        #the rest of them are simply hardcoded with the major category
        data = np.array([vehicles_involved,casualties,hour,day_of_week,driving_experience,4,0,2,driver_age,0,accident_cause,
                         vehicle_type,accident_area,vehicle_movement,road_alligment,type_of_collision,3,0,5,
                         lanes,0,0,0,1,0,1,0]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)
        st.write('<hr>', unsafe_allow_html=True)
        st.write(f"The predicted severity is:  {lstAccidentSeverity[pred[0]-1]}")

if __name__ == '__main__':
    main()