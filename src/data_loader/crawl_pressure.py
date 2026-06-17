import cdsapi
import os

def _submit_jobs():
    client = cdsapi.Client()
    jobs = []

    for year in ['2025']:

        dataset = "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "geopotential",
                "relative_humidity",
                "u_component_of_wind",
                "v_component_of_wind",
                'vertical_velocity'
            ],
            "year": [year],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "06:00", "12:00",
                "18:00"
            ],
            "pressure_level": ["500", "850", "200"],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": [60, 100, 5, 180]
        }   

        r = client.retrieve(dataset, request)
        jobs.append((year, r))

    return jobs


def _download_jobs(jobs):
    for year, r in jobs:

        output_path = f"data.grib"

        print(f"⬇️ Downloading {year}...")
        r.download(output_path)

def crawl_pressure():
    jobs = _submit_jobs()
    _download_jobs(jobs)

if __name__ == '__main__':
    crawl_pressure()