import json
import cv2
from utils.helper_utils import prepare_folder

class VideoOperation:
    def __init__(self):
        self.fps = json.load(open('config/settings.json'))["fps"]
        self.input_file_with_path = json.load(open('config/settings.json'))["input"]
        self.output_file_with_path = json.load(open('config/settings.json'))["output"]
        self.converted_frames_folder = json.load(open('config/settings.json'))["convert_frame_folder"]
        self.proccessed_frames_folder = json.load(open('config/settings.json'))["proccessed_frame_folder"]


    def extract_frames(self):
        cap = cv2.VideoCapture(self.input_file_with_path)

        original_fps = cap.get(cv2.CAP_PROP_FPS)

        frame_interval = int(original_fps / self.fps)
        
        # Create an output directory from converted frames
        prepare_folder(self.converted_frames_folder)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only save frames at the specified interval
            if frame_count % frame_interval == 0:
                output_file = f"{self.converted_frames_folder}/frame_{extracted_count}.jpg"
                cv2.imwrite(output_file, frame)
                # print(f"Frame {extracted_count} has been extracted and saved as {output_file}")
                extracted_count += 1
            
            frame_count += 1
        
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

    