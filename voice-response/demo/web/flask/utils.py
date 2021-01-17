import os
from playsound import playsound
import glob

RESPONSES_DIR = r"C:\Users\phalc\Documents\Dao-Tao\Course-Deep-Learning\Project\Cuoi-Ky\dl-end-term\voice-response\demo\responses-paths"

# PREPARE THE RESPONSE PATHS
def response_paths(facialExpressionPrediction):
    files_path = os.path.join(RESPONSES_DIR, f'{facialExpressionPrediction}_paths.txt')
    with open(files_path) as f:
        lines = [line.rstrip() for line in f]
    return lines
