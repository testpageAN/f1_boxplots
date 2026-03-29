def extract_race(pathname):
    result = pathname.split("/")[-1].split("_")[2]
    return result

