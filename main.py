from fastapi import FastAPI, UploadFile, File, Request,Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import base64
from pydantic import BaseModel
from typing import Annotated
from datetime import date, datetime, timedelta
import json
import os, pickle, tempfile, pytz, subprocess, re, io, time
import cv2
# import face_recognition
from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import bigquery
from code import download_blob
from tempfile import TemporaryFile

bucket_name = 'emp_png'
key_path = "cloudkarya-internship-415b6b4ef0ff.json"
client = storage.Client.from_service_account_json(key_path)  
bucket = client.get_bucket(bucket_name)
bigquery_client = bigquery.Client.from_service_account_json(key_path)
storage_client = storage.Client.from_service_account_json(key_path)
PROJECT_ID = "cloudkarya-internship"
DATASET_ID = "eams1"  # Remove any whitespaces around the dataset ID
TABLE_ID = "ImageDataTable" 

def extract(request: Request):
    download_blob(bucket_name, source_file_name, dest_filename)

def list_images(bucket_name):
    blobs = client.list_blobs(bucket_name)
    images = []
    for blob in blobs:
        image_path = download_blob(bucket_name, blob.name, blob.name)
        images.append(image_path)
    return images


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get('/')  
def index(request : Request):
    context={"request" : request,
             "predictedtopic":"No Video"}
    return templates.TemplateResponse("index.html",context) 

@app.get("/main", response_class=HTMLResponse)
def lis( request : Request):
    images = list_images(bucket_name)  
    print(images)
    context = {"request": request, "images": images}
    return templates.TemplateResponse("index.html", context)    
# #Upload video button ver
# # @app.post("/upload_video", response_class=HTMLResponse)
# # async def upload_video(request : Request, video_file: UploadFile = File(...)):
# #     video_path = f"videos/{video_file.filename}"
# #     with open(video_path,"wb") as f:
# #         f.write(await video_file.read())
 
# #     a=extract_frames(video_path)   
# #     b=recognize_faces(a)
# #     #c=process_attendance_data(b)
# #     context = {
# #         "request": request, 
# #         "video_path": video_path,
# #         "b": b
# #     }
# #     return templates.TemplateResponse("index.html",context)

# #To download model.pkl .. which is present but don't delete .. just in case
# # def download_blob(bucket_name, source_file_name, dest_filename,storage_client):
# #     bucket = storage_client.get_bucket(bucket_name)
# #     blob = bucket.blob(source_file_name)
# #     f = open(dest_filename,'wb')
# #     blob.download_to_file(f)

# #download_blob("emp_monitoring_videos_raw", "cloudkarya/model.pkl", "model.pkl",storage_client=client) 

# with open('model.pkl', 'rb') as f:
#     known_faces, known_names = pickle.load(f)
  

# def extract_frames(video_path):
#     print(f"Video = {video_path}")
#     count = 0
#     cap = cv2.VideoCapture(video_path)

#     frame_counter = 0
#     frames = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         count += 1

#         if count % 20 != 0:
#             continue

#         frames.append(frame)

#     cap.release()
#     return frames

# def convert_frames_to_jpeg(frames):
#     converted_frames = []

#     for frame in frames:
#         # Convert the frame array to PIL format
#         pil_image = Image.fromarray(frame)

#         # Convert the PIL image to JPEG format and get the bytes
#         with io.BytesIO() as output:
#             pil_image.save(output, format='JPEG')
#             jpeg_bytes = output.getvalue()

#         # Append the converted frame to the list
#         converted_frames.append(jpeg_bytes)

#     return converted_frames

# def match_video_time_created(output_str):
#     # Use regular expressions to extract the creation_time
#     match = re.search(r'creation_time\s+:\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', output_str)
#     if match:
#         creation_date_time = match.group(1)
#         print(creation_date_time)
#         date_format = "%Y-%m-%dT%H:%M:%S"
#     # Convert the string to a datetime object
#         creation_date_time = datetime.datetime.strptime(creation_date_time, "%Y-%m-%dT%H:%M:%S")
#         print(creation_date_time)
#         i = pytz.timezone('Asia/Kolkata')
#         creation_date_time = creation_date_time.astimezone(i)
#         return creation_date_time


# def recognize_faces(frame, download_video, video_created_time, event=None, context=None):

# #   if event == None:
# #       frame ='frame_11.jpg'
# #   else:
# #       frame = event['name']
# ##Image from storage-downloaded
# #   frame_video = frame.split("/")[-1]
# #   download_blob("emp_attendance_monitoring_processed", frame, frame_video, storage_client=client)


# #   frame = cv2.imread(frame_video)## open frame from storage
#   cap = cv2.VideoCapture(download_video)

#   attendance_dict = {}  # Dictionary to store attendance data

#   # Get the original frame size 
#   width = frame.shape[1]
#   height = frame.shape[0]

#   # Calculate the cropping coordinates
#   crop_x = (width - min(width, height)) // 2
#   crop_y = (height - min(width, height)) // 2
#   crop_width = min(width, height)
#   crop_height = min(width, height)

#   # Desired square frame size
#   square_size = 500

#   # Crop and resize frame
#   cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
#   resized_frame = cv2.resize(cropped_frame, (square_size, square_size))

#   # Find faces in the frame
#   face_locations = face_recognition.face_locations(resized_frame)
#   face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

#   # Iterate over each detected face
#   for face_encoding, face_location in zip(face_encodings, face_locations):
#     # Compare face encoding with the known faces
#     matches = face_recognition.compare_faces(known_faces, face_encoding)
#     name = "Unknown"

#     # Find the best match
#     if len(matches) > 0:
#       face_distances = face_recognition.face_distance(known_faces, face_encoding)
#       best_match_index = np.argmin(face_distances)
#       if matches[best_match_index]:
#         name = known_names[best_match_index]
#         # Update attendance dictionary with name and timestamp
#         # attendance_dict[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#       # Draw a box around the face and label the name
#       if face_locations:
#         timestamp = cap.get(round(cv2.CAP_PROP_POS_MSEC,2)) / 1000.0
#         adjusted_timestamp = video_created_time + datetime.timedelta(seconds=timestamp)
#         attendance_dict[name] = adjusted_timestamp.strftime("%Y-%m-%d %H:%M:%S")
#       top, right, bottom, left = face_location
#       cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
#       cv2.putText(resized_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#       cv2.putText(resized_frame, str(adjusted_timestamp.strftime("%Y-%B-%d %H:%M:%S")), (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#   # Display the resulting frame
#     plt.imshow(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
#     print (attendance_dict)
#   #Upload to Big Query
#     global PROJECT_ID
#     global DATASET_ID
#     global TABLE_ID
#     client1 = bigquery.Client().from_service_account_json("keys.json")
#     rows_to_insert = []
#     for name, timestamp in attendance_dict.items():
#       dt_str, time_str = timestamp.split(' ') #dt_p = 
#       # dt_str = dt_p[0]
#       # time_str = dt_p[1]
#       en_ex = "Entry"
#       row_dict = {
#         "Name": name,
#         "Date": dt_str,
#         "Time": time_str,
#         "EntryExit": en_ex
#       }
#       rows_to_insert.append(row_dict)

#     # Insert the rows into the BigQuery table
#     errors = client1.insert_rows_json(f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}", rows_to_insert)
#     if errors == []:
#         print("New rows have been added.")
#     else:
#         print("Encountered errors while inserting rows: {}".format(errors))
#   #return attendance_dict
#     ## saved to BQ


# def process_file():
#     """Triggered by a change to a Cloud Storage bucket.
#     Args:
#          event (dict): Event payload.
#          context (google.cloud.functions.Context): Metadata for the event.
#     """
#     # if event == None:
#     #     file_name='cloudkarya/cloudkarya_Trail.mp4'
#     # else:
#     #     file_name = event['name']
#     file_name='cloudkarya/cloudkarya_Trail.mp4'
#     print(f"Processing file: {file_name}.")

#     storage_client = storage.Client()

#     source_bucket = storage_client.bucket("emp_monitoring_videos_raw")
#     source_blob = source_bucket.blob(file_name)
#     destination_bucket = client.bucket("emp_attendance_monitoring_processed")

#     download_video = file_name.split("/")[-1]
#     download_blob("emp_monitoring_videos_raw", file_name, download_video, storage_client=client)

#     command = f"ffmpeg -i '/content/{download_video}' -dump"
#     process = subprocess.run(command, shell=True, capture_output=True, text=True)
#     # Check the exit code of the process
#     if process.returncode == 0:
#         # Command executed successfully
#         output = process.stdout
#         video_created_date_time = match_video_time_created(output)
#         date_str = datetime.datetime.strftime(video_created_date_time, "%Y-%m-%d")
#         print(video_created_date_time)
#     else:
#         # Command encountered an error but still works
#         error = process.stderr
#         video_created_date_time = match_video_time_created(error)
#         date_str = datetime.datetime.strftime(video_created_date_time, "%Y-%m-%d")
#         print(video_created_date_time)

#     # Create a temporary directory to store the frames
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # Extract frames from the video file and save them as images in the temporary directory
#         frames = extract_frames(download_video)

#         frames_len = len(frames)
#         print(f"Number of frames = {frames_len}")

#         # Write the extracted frames to a new file in the destination bucket.
#         frame_counter = 0
#         for frame in frames:
#             # Save the frame as a local image file
#             # frame_filename = f"frame_{frame_counter}.jpg" #frame_1
#             frame_filename = f"frame_{date_str}_{frame_counter}.jpg"  #frame_2023-06-07_1
#             local_image_path = os.path.join(tmpdir, frame_filename)
#             cv2.imwrite(local_image_path, frame)
#             # Upload the local image file to the destination bucket
#             destination_blob = destination_bucket.blob(frame_filename)
#             destination_blob.upload_from_filename(local_image_path)
#             frame_counter += 1
#             print(f'Frame_{date_str}_{frame_counter} sent')

#             recog = recognize_faces(frame, download_video, video_created_date_time)


# # Call the function to trigger the processing .. got error
# # process_file()

# # def recognize_faces(frames):
# #     attendance_dict = {}  # Dictionary to store attendance data

# #     for i, frame in enumerate(frames):
# #         # Get the original frame size
# #         width = frame.shape[1]
# #         height = frame.shape[0]

# #         # Calculate the cropping coordinates
# #         crop_x = (width - min(width, height)) // 2
# #         crop_y = (height - min(width, height)) // 2
# #         crop_width = min(width, height)
# #         crop_height = min(width, height)

# #         # Desired square frame size
# #         square_size = 500

# #         # Crop and resize frame
# #         cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
# #         resized_frame = cv2.resize(cropped_frame, (square_size, square_size))

# #         # Find faces in the frame
# #         face_locations = face_recognition.face_locations(resized_frame)
# #         face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

# #         if len(face_locations) == 0:
# #             # Skip the frame if no faces are detected
# #             continue

# #         # Iterate over each detected face
# #         for face_encoding, face_location in zip(face_encodings, face_locations):
# #             # Compare face encoding with the known faces
# #             matches = face_recognition.compare_faces(known_faces, face_encoding)
# #             name = "Unknown"

# #             # Find the best match
# #             if len(matches) > 0:
# #               face_distances = face_recognition.face_distance(known_faces, face_encoding)
# #               best_match_index = np.argmin(face_distances)
# #               if matches[best_match_index]:
# #                   name = known_names[best_match_index]
# #                   # Update attendance dictionary with name and timestamp
# #                   # attendance_dict[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# #               # Draw a box around the face and label the name
# #               if face_locations:
# #                 timestamp = cap.get(round(cv2.CAP_PROP_POS_MSEC,2)) / 1000.0
# #                 adjusted_timestamp = video_created_time + datetime.timedelta(seconds=timestamp)
# #                 attendance_dict[name] = adjusted_timestamp.strftime("%Y-%B-%d %H:%M:%S")
# #               top, right, bottom, left = face_location
# #               cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
# #               cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# #               cv2.putText(frame, str(adjusted_timestamp.strftime("%Y-%B-%d %H:%M:%S")), (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
# #         # Save the resulting frame as an image
# #         output_path = f'results/frame_{i}.jpg'
# #         cv2.imwrite(output_path, resized_frame) 
# #         html_table = "<table>\n"
# #         html_table += "<tr><th colspan='3' style='text-align: center;'>Attendance</th></tr>\n"
# #         html_table += "<tr><th>Name</th><th>Date</th><th>Time</th></tr>\n"
# #         html_table += "</thead>\n" 
# #         for name, date in attendance_dict.items():
# #             date_parts = date.split(' ')
# #             date_str = date_parts[0]
# #             time_str = date_parts[1]
# #             html_table += f"<tr><td>{name}</td><td>{date_str}</td><td>{time_str}</td></tr>\n"
  
# #         html_table += "</table>"   
# #     return html_table                

#Retrieve from BIG QUERY   
@app.get("/action_page") 
async def get_data(request: Request, choose_date: str):
    global PROJECT_ID
    global DATASET_ID
    global TABLE_ID

    # Define the time difference threshold in minutes
    time_threshold_minutes = 10

    # Initialize variables to keep track of the previous values
    prev_name = None
    prev_time = None
    prev_entry_exit = None

    # Initialize a list to store the selected records
    selected_records = []

    query = f"""
         SELECT Name, Date, Time, EntryExit FROM {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}
         WHERE Date = '{choose_date}' AND Company = 'CloudKarya'
         ORDER BY Time ASC;"""

    df = bigquery_client.query(query).to_dataframe() 

    for _, row in df.iterrows():
        name = row['Name']
        time = datetime.strptime(row['Time'], "%H:%M:%S")
        entry_exit = row['EntryExit']

        # Check if the current Name is different from the previous one,
        # or if the time difference is greater than or equal to the threshold,
        # or if the EntryExit value is different
        if (name != prev_name or
            (prev_time and (time - prev_time) >= timedelta(minutes=time_threshold_minutes)) or
            entry_exit != prev_entry_exit):

            # Add the current row as a dictionary to the selected_records list
            selected_records.append({
                'Name': name,
                'Date': row['Date'],
                'Time': row['Time'],
                'EntryExit': entry_exit
            })

        prev_name = name
        prev_time = time
        prev_entry_exit = entry_exit

    return templates.TemplateResponse(
        'index.html',
        context={"request": request, "attendance_df": selected_records, "chosen_date": choose_date}
    ) 
##Previous query
# async def get_data(request: Request, choose_date : str):
#     global PROJECT_ID
#     global DATASET_ID
#     global TABLE_ID
#     query = f"""
#          SELECT  * FROM {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}
#          WHERE date ='{choose_date}' ORDER BY Time ASC;"""
#     df = bigquery_client.query(query).to_dataframe()
#     df = df.to_dict(orient='records')
#     return templates.TemplateResponse('index.html', context={"request": request ,"attendance_df" : df, "chosen_date" : choose_date})







# def process_attendance_data(attendance_dict):
#     # Convert the att endance dictionary to a DataFrame
#     df = pd.DataFrame.from_dict(attendance_dict, orient='index', columns=['Timestamp'])

#     # Split timestamp into separate date and time columns
#     df[['Date', 'Time']] = df['Timestamp'].str.split(' ', 1, expand=True)

#     # Remove the original timestamp column
#     df = df.drop("Timestamp", axis=1)

#     # Set the Entry/Exit column as 'Entry'
#     df['Entry/Exit'] = 'Entry'
#     df = df.sort_values("Time")
#     current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_file = f'Attendance_{current_datetime}.csv'
#     df.to_csv(output_file, index=True)
#     return output_file





 