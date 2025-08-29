from datetime import datetime
from astroquery.mast import Observations

#datatime to MJD conversion function
def datetime_to_mjd(dt):
    """Convert a datetime object to Modified Julian Date (MJD)."""
    return (dt - datetime(1858, 11, 17, 0, 0, 0)).days + 2400000.5

# Printing all missions
print(Observations.list_missions())

# Getting the light curve data for objects in the TESS mission
start_date = datetime_to_mjd(datetime(2025, 1, 1))
end_date = datetime_to_mjd(datetime.now())
tess_lightcurves = Observations.query_criteria(obs_collection="TESS", target_classification="star", t_min=start_date, t_max=end_date)