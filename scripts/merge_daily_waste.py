import pandas as pd
import glob

DAILY_WASTE_FILE_PATH_PATTERN = '../dataset/daily_waste/*.csv'
DAILY_WASTE_OUTPUT_FILE_PATH = '../tmp/daily_waste_combined.csv'

if __name__ == '__main__':
    data_frames = []
    for file_path in glob.glob(DAILY_WASTE_FILE_PATH_PATTERN):
        temp_df = pd.read_csv(file_path)
        data_frames.append(temp_df)

    daily_waste_df = pd.concat(data_frames)

    daily_waste_df.to_csv(DAILY_WASTE_OUTPUT_FILE_PATH, index=False)
