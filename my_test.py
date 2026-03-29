import os

def extract_race(pathname):
    result = pathname.split("/")[-1].split("_")[2]
    return result

root_directory = os.path.dirname(__file__)
charts_directory = os.path.join(root_directory, "charts_plots")
# print(charts_directory)
