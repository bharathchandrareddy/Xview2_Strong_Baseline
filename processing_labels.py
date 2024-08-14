import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from shapely import wkt
from shapely.geometry import Polygon
from geopy.geocoders import Nominatim

# Define location of the images and labels
xbd_dir = "C:\\Users\\PC\\Desktop\\damage_assessement_data"
train_dir = os.path.join(xbd_dir, 'train')
tier3 = os.path.join(xbd_dir, 'tier3')
images_dir = os.path.join(xbd_dir, "original_images")
labels_dir = os.path.join(xbd_dir, "labels")
masks_dir = os.path.join(xbd_dir, 'masks')
if not os.path.isdir(masks_dir):
    os.makedirs(masks_dir)
train_data_info = os.path.join(xbd_dir, "train_data_info.csv")
list_of_dirs = ['train', 'tier3']

# Create DataFrame
columns = ['pre_image_name', 'post_image_name', 'disaster_name', 'disaster_type',
           'disaster_location', 'num_buildings_pre', 'num_buildings_post',
           'num_no_damage', 'num_minor_damage', 'num_major_damage',
           'num_destroyed', 'num_un_classified', 'pre_disaster_capture_date', 'post_disaster_capture_date']
df = pd.DataFrame(columns=columns)

# Dictionary to cache disaster locations to avoid repeated geolocation lookups
disaster_location_cache = {}

# Function to get disaster location (this could be further optimized or replaced with a local database)
def get_disaster_location(lang_lat_features, disaster_name):
    if disaster_name in disaster_location_cache:
        return disaster_location_cache[disaster_name]
    
    if len(lang_lat_features) > 0:
        location_coords = lang_lat_features[0]["wkt"]
        centroid = Polygon(wkt.loads(location_coords)).centroid
        # Here, instead of making a call to an external API, use a simple mapping, database lookup, or offline geocoder.
        # Placeholder example:
        #get location of the centroid
        geolocator = Nominatim(user_agent='geo_locator')
        location = geolocator.reverse((centroid.y, centroid.x))
        # get state/city and country from location
        if 'state' in location.raw['address']:
            disaster_location = location.raw['address']['state'] + ", " + location.raw['address']['country']
        elif 'city' in location.raw['address']:
            disaster_location = location.raw['address']['city'] + ", " + location.raw['address']['country']
        else:
            disaster_location = 'None'
        disaster_location_cache[disaster_name] = disaster_location
        # disaster_location = f"Lat: {centroid.y}, Lon: {centroid.x}"  # Replace with a more accurate method.
        # disaster_location_cache[disaster_name] = disaster_location
        return disaster_location
    return None

# Iterate through all labels files
for directory in list_of_dirs:
    dir_path = os.path.join(xbd_dir, directory)
    labels_dir = os.path.join(dir_path, 'labels')
    for json_filename in tqdm(os.listdir(labels_dir)):
        # Only process pre-disaster JSON files
        if json_filename.endswith('.json') and "_pre" in json_filename:
            # Read JSON file
            pre_json_filepath = os.path.join(labels_dir, json_filename)
            post_json_filepath = os.path.join(labels_dir, json_filename.replace("_pre", "_post"))
            if not os.path.exists(post_json_filepath):
                continue
            
            # Read pre-disaster JSON
            with open(pre_json_filepath, 'r') as file:
                pre_data = json.load(file)
                pre_image_name = pre_data["metadata"]["img_name"]
                disaster_name = pre_data["metadata"]["disaster"]
                disaster_type = pre_data["metadata"]["disaster_type"]
                pre_disaster_capture_date = datetime.fromisoformat(pre_data["metadata"]["capture_date"][:-1]).date()
                lang_lat_features = pre_data["features"]["lng_lat"]
                num_buildings_pre = len(lang_lat_features)
                disaster_location = get_disaster_location(lang_lat_features, disaster_name)

            # Read post-disaster JSON
            with open(post_json_filepath, 'r') as file:
                post_data = json.load(file)
                post_image_name = post_data["metadata"]["img_name"]
                post_disaster_capture_date = datetime.fromisoformat(post_data["metadata"]["capture_date"][:-1]).date()
                building_features = post_data["features"]["xy"]
                num_buildings_post = len(building_features)
                
                # Count number of buildings by damage type
                building_damage_info = {"no-damage": 0, "minor-damage": 0, "major-damage": 0, "destroyed": 0, "un-classified": 0}
                for feature in building_features:
                    damage_type = feature["properties"]["subtype"]
                    building_damage_info[damage_type] += 1
                
                num_no_damage = building_damage_info["no-damage"]
                num_minor_damage = building_damage_info["minor-damage"]
                num_major_damage = building_damage_info["major-damage"]
                num_destroyed = building_damage_info["destroyed"]
                num_un_classified = building_damage_info["un-classified"]
            
            # Create DataFrame row
            row_data = {
                "pre_image_name": pre_image_name,
                "post_image_name": post_image_name,
                "disaster_name": disaster_name,
                "disaster_type": disaster_type,
                "disaster_location": disaster_location,
                "num_buildings_pre": num_buildings_pre,
                "num_buildings_post": num_buildings_post,
                "num_no_damage": num_no_damage,
                "num_minor_damage": num_minor_damage,
                "num_major_damage": num_major_damage,
                "num_destroyed": num_destroyed,
                "num_un_classified": num_un_classified,
                "pre_disaster_capture_date": pre_disaster_capture_date,
                "post_disaster_capture_date": post_disaster_capture_date
            }
            
            # Append row to DataFrame
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

# Save DataFrame to CSV
df.to_csv(train_data_info, index=False)
