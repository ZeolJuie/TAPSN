import copy
import json
import logging
import random

import carla


REGIME_DEFINITIONS = {
    'clear_day': {
        'category': 'day',
        'sun_altitude_range': [38.0, 75.0],
        'cloudiness_range': [0.0, 20.0],
        'precipitation_probability': 0.0,
        'precipitation_range': [0.0, 0.0],
        'post_rain_probability': 0.01,
        'fog_probability': 0.0,
        'fog_density_range': [0.0, 2.0],
        'fog_distance_range': [250.0, 1000.0],
        'wind_range': [3.0, 15.0],
        'scattering_range': [0.45, 0.70],
        'mie_range': [0.04, 0.08],
        'rayleigh_range': [0.06, 0.09],
        'min_brightness': 0.48,
    },
    'partly_cloudy_day': {
        'category': 'day',
        'sun_altitude_range': [32.0, 70.0],
        'cloudiness_range': [20.0, 55.0],
        'precipitation_probability': 0.04,
        'precipitation_range': [1.0, 5.0],
        'post_rain_probability': 0.04,
        'fog_probability': 0.02,
        'fog_density_range': [2.0, 6.0],
        'fog_distance_range': [160.0, 600.0],
        'wind_range': [4.0, 16.0],
        'scattering_range': [0.40, 0.65],
        'mie_range': [0.04, 0.08],
        'rayleigh_range': [0.06, 0.09],
        'min_brightness': 0.42,
    },
    'bright_overcast_day': {
        'category': 'day',
        'sun_altitude_range': [28.0, 62.0],
        'cloudiness_range': [60.0, 88.0],
        'precipitation_probability': 0.14,
        'precipitation_range': [1.0, 10.0],
        'post_rain_probability': 0.10,
        'fog_probability': 0.03,
        'fog_density_range': [3.0, 8.0],
        'fog_distance_range': [120.0, 450.0],
        'wind_range': [4.0, 18.0],
        'scattering_range': [0.38, 0.62],
        'mie_range': [0.035, 0.075],
        'rayleigh_range': [0.06, 0.085],
        'min_brightness': 0.34,
    },
    'light_rain_day': {
        'category': 'day',
        'sun_altitude_range': [20.0, 48.0],
        'cloudiness_range': [75.0, 100.0],
        'precipitation_probability': 1.0,
        'precipitation_range': [8.0, 30.0],
        'post_rain_probability': 0.0,
        'fog_probability': 0.02,
        'fog_density_range': [2.0, 7.0],
        'fog_distance_range': [140.0, 420.0],
        'wind_range': [8.0, 22.0],
        'scattering_range': [0.34, 0.58],
        'mie_range': [0.035, 0.07],
        'rayleigh_range': [0.06, 0.085],
        'min_brightness': 0.23,
    },
    'misty_morning': {
        'category': 'day',
        'sun_altitude_range': [10.0, 24.0],
        'cloudiness_range': [20.0, 75.0],
        'precipitation_probability': 0.04,
        'precipitation_range': [1.0, 5.0],
        'post_rain_probability': 0.18,
        'fog_probability': 0.75,
        'fog_density_range': [8.0, 24.0],
        'fog_distance_range': [55.0, 220.0],
        'wind_range': [2.0, 10.0],
        'scattering_range': [0.32, 0.58],
        'mie_range': [0.04, 0.085],
        'rayleigh_range': [0.06, 0.085],
        'min_brightness': 0.14,
    },
    'sunset_clear': {
        'category': 'sunset',
        'sun_altitude_range': [4.0, 14.0],
        'cloudiness_range': [0.0, 35.0],
        'precipitation_probability': 0.0,
        'precipitation_range': [0.0, 0.0],
        'post_rain_probability': 0.03,
        'fog_probability': 0.02,
        'fog_density_range': [2.0, 5.0],
        'fog_distance_range': [160.0, 600.0],
        'wind_range': [2.0, 14.0],
        'scattering_range': [0.42, 0.72],
        'mie_range': [0.045, 0.09],
        'rayleigh_range': [0.065, 0.09],
        'min_brightness': 0.28,
    },
    'sunset_cloudy': {
        'category': 'sunset',
        'sun_altitude_range': [5.0, 16.0],
        'cloudiness_range': [35.0, 75.0],
        'precipitation_probability': 0.08,
        'precipitation_range': [1.0, 8.0],
        'post_rain_probability': 0.08,
        'fog_probability': 0.03,
        'fog_density_range': [2.0, 6.0],
        'fog_distance_range': [140.0, 500.0],
        'wind_range': [3.0, 16.0],
        'scattering_range': [0.36, 0.64],
        'mie_range': [0.04, 0.085],
        'rayleigh_range': [0.06, 0.085],
        'min_brightness': 0.24,
    },
    'sunset_overcast': {
        'category': 'sunset',
        'sun_altitude_range': [8.0, 18.0],
        'cloudiness_range': [65.0, 90.0],
        'precipitation_probability': 0.12,
        'precipitation_range': [1.0, 10.0],
        'post_rain_probability': 0.10,
        'fog_probability': 0.02,
        'fog_density_range': [2.0, 5.0],
        'fog_distance_range': [150.0, 500.0],
        'wind_range': [4.0, 18.0],
        'scattering_range': [0.34, 0.58],
        'mie_range': [0.035, 0.075],
        'rayleigh_range': [0.06, 0.085],
        'min_brightness': 0.22,
    },
    'blue_hour': {
        'category': 'sunset',
        'sun_altitude_range': [-3.0, 4.0],
        'cloudiness_range': [0.0, 60.0],
        'precipitation_probability': 0.04,
        'precipitation_range': [1.0, 5.0],
        'post_rain_probability': 0.05,
        'fog_probability': 0.03,
        'fog_density_range': [2.0, 5.0],
        'fog_distance_range': [160.0, 600.0],
        'wind_range': [2.0, 14.0],
        'scattering_range': [0.30, 0.54],
        'mie_range': [0.035, 0.075],
        'rayleigh_range': [0.06, 0.08],
        'min_brightness': 0.16,
    },
    'urban_night_clear': {
        'category': 'night',
        'sun_altitude_range': [-12.0, -4.0],
        'cloudiness_range': [0.0, 35.0],
        'precipitation_probability': 0.0,
        'precipitation_range': [0.0, 0.0],
        'post_rain_probability': 0.02,
        'fog_probability': 0.0,
        'fog_density_range': [0.0, 2.0],
        'fog_distance_range': [250.0, 1000.0],
        'wind_range': [2.0, 12.0],
        'scattering_range': [0.28, 0.48],
        'mie_range': [0.035, 0.07],
        'rayleigh_range': [0.06, 0.08],
        'min_brightness': 0.10,
    },
    'urban_night_cloudy': {
        'category': 'night',
        'sun_altitude_range': [-10.0, -3.0],
        'cloudiness_range': [35.0, 80.0],
        'precipitation_probability': 0.04,
        'precipitation_range': [1.0, 5.0],
        'post_rain_probability': 0.06,
        'fog_probability': 0.01,
        'fog_density_range': [2.0, 4.0],
        'fog_distance_range': [180.0, 600.0],
        'wind_range': [3.0, 14.0],
        'scattering_range': [0.26, 0.46],
        'mie_range': [0.035, 0.07],
        'rayleigh_range': [0.06, 0.08],
        'min_brightness': 0.08,
    },
    'wet_night': {
        'category': 'night',
        'sun_altitude_range': [-8.0, -3.0],
        'cloudiness_range': [60.0, 95.0],
        'precipitation_probability': 0.25,
        'precipitation_range': [1.0, 15.0],
        'post_rain_probability': 0.85,
        'fog_probability': 0.01,
        'fog_density_range': [1.0, 4.0],
        'fog_distance_range': [180.0, 600.0],
        'wind_range': [4.0, 16.0],
        'scattering_range': [0.25, 0.44],
        'mie_range': [0.035, 0.07],
        'rayleigh_range': [0.06, 0.08],
        'min_brightness': 0.07,
    },
    'rainy_night': {
        'category': 'night',
        'sun_altitude_range': [-8.0, -2.0],
        'cloudiness_range': [85.0, 100.0],
        'precipitation_probability': 1.0,
        'precipitation_range': [15.0, 45.0],
        'post_rain_probability': 0.0,
        'fog_probability': 0.0,
        'fog_density_range': [0.0, 2.0],
        'fog_distance_range': [220.0, 800.0],
        'wind_range': [8.0, 24.0],
        'scattering_range': [0.25, 0.42],
        'mie_range': [0.035, 0.065],
        'rayleigh_range': [0.06, 0.08],
        'min_brightness': 0.045,
    },
}

OFFICIAL_PRESET_FIELDS = {
    'ClearNight': {
        'cloudiness': 5.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -90.0,
        'fog_density': 60.0,
        'fog_distance': 75.0,
        'wetness': 0.0,
        'fog_falloff': 1.0,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'ClearNoon': {
        'cloudiness': 5.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 2.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'ClearSunset': {
        'cloudiness': 5.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 2.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'CloudyNight': {
        'cloudiness': 60.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -90.0,
        'fog_density': 60.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'CloudyNoon': {
        'cloudiness': 60.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 3.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'CloudySunset': {
        'cloudiness': 60.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 3.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'Default': {
        'cloudiness': -1.0,
        'precipitation': -1.0,
        'precipitation_deposits': -1.0,
        'wind_intensity': -1.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -1.0,
        'fog_density': -1.0,
        'fog_distance': -1.0,
        'wetness': -1.0,
        'fog_falloff': -1.0,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'DustStorm': {
        'cloudiness': 100.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 100.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 2.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 100.0,
    },
    'HardRainNight': {
        'cloudiness': 100.0,
        'precipitation': 100.0,
        'precipitation_deposits': 90.0,
        'wind_intensity': 100.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -90.0,
        'fog_density': 100.0,
        'fog_distance': 0.75,
        'wetness': 100.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'HardRainNoon': {
        'cloudiness': 100.0,
        'precipitation': 100.0,
        'precipitation_deposits': 90.0,
        'wind_intensity': 100.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 7.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'HardRainSunset': {
        'cloudiness': 100.0,
        'precipitation': 100.0,
        'precipitation_deposits': 90.0,
        'wind_intensity': 100.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 7.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'MidRainSunset': {
        'cloudiness': 60.0,
        'precipitation': 60.0,
        'precipitation_deposits': 60.0,
        'wind_intensity': 60.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 3.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'MidRainyNight': {
        'cloudiness': 80.0,
        'precipitation': 60.0,
        'precipitation_deposits': 60.0,
        'wind_intensity': 60.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -90.0,
        'fog_density': 60.0,
        'fog_distance': 0.75,
        'wetness': 80.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'MidRainyNoon': {
        'cloudiness': 60.0,
        'precipitation': 60.0,
        'precipitation_deposits': 60.0,
        'wind_intensity': 60.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 3.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'SoftRainNight': {
        'cloudiness': 60.0,
        'precipitation': 30.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 30.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -90.0,
        'fog_density': 60.0,
        'fog_distance': 0.75,
        'wetness': 60.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'SoftRainNoon': {
        'cloudiness': 20.0,
        'precipitation': 30.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 30.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 3.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'SoftRainSunset': {
        'cloudiness': 20.0,
        'precipitation': 30.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 30.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 2.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'WetCloudyNight': {
        'cloudiness': 60.0,
        'precipitation': 0.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -90.0,
        'fog_density': 60.0,
        'fog_distance': 0.75,
        'wetness': 60.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'WetCloudyNoon': {
        'cloudiness': 60.0,
        'precipitation': 0.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 3.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'WetCloudySunset': {
        'cloudiness': 60.0,
        'precipitation': 0.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 2.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'WetNight': {
        'cloudiness': 5.0,
        'precipitation': 0.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': -90.0,
        'fog_density': 60.0,
        'fog_distance': 75.0,
        'wetness': 60.0,
        'fog_falloff': 1.0,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'WetNoon': {
        'cloudiness': 5.0,
        'precipitation': 0.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 3.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
    'WetSunset': {
        'cloudiness': 5.0,
        'precipitation': 0.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': -1.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 2.0,
        'fog_distance': 0.75,
        'wetness': 0.0,
        'fog_falloff': 0.1,
        'scattering_intensity': 1.0,
        'mie_scattering_scale': 0.03,
        'rayleigh_scattering_scale': 0.0331,
        'dust_storm': 0.0,
    },
}

DEFAULT_REGIME_WEIGHTS = {
    'ClearNoon': 0.18,
    'CloudyNoon': 0.14,
    'WetNoon': 0.08,
    'WetCloudyNoon': 0.08,
    'SoftRainNoon': 0.09,
    'MidRainyNoon': 0.07,
    'HardRainNoon': 0.05,
    'DustStorm': 0.01,
    'ClearSunset': 0.04,
    'CloudySunset': 0.035,
    'WetSunset': 0.025,
    'WetCloudySunset': 0.02,
    'SoftRainSunset': 0.02,
    'MidRainSunset': 0.007,
    'HardRainSunset': 0.003,
    'ClearNight': 0.025,
    'CloudyNight': 0.025,
    'WetNight': 0.02,
    'WetCloudyNight': 0.02,
    'SoftRainNight': 0.025,
    'MidRainyNight': 0.025,
    'HardRainNight': 0.01,
}

DEFAULT_WEATHER_CONFIG = {
    'enabled': True,
    'seed': None,
    'regimes': {
        'weights': DEFAULT_REGIME_WEIGHTS,
    },
    'realism': {
        'max_resample_attempts': 8,
        'min_scattering_intensity': 0.25,
        'min_mie_scattering_scale': 0.03,
        'min_rayleigh_scattering_scale': 0.06,
    },
    'preset_jitter': {
        'enabled': True,
        'scale': 1.0,
        'cloudiness_std': 6.0,
        'precipitation_std': 6.0,
        'precipitation_deposits_std': 8.0,
        'wind_intensity_std': 6.0,
        'wetness_std': 8.0,
        'sun_altitude_std': 3.0,
    },
    'sun_altitude': {
        'weights': {
            'day': 0.55,
            'low_sun': 0.15,
            'twilight': 0.15,
            'night': 0.15,
        },
        'day_range': [25.0, 75.0],
        'low_sun_range': [8.0, 25.0],
        'twilight_range': [-8.0, 8.0],
        'night_range': [-75.0, -8.0],
    },
    'sun_azimuth': {
        'range': [0.0, 360.0],
    },
    'cloudiness': {
        'weights': {
            'clear': 0.20,
            'mid': 0.50,
            'overcast': 0.30,
        },
        'clear_range': [0.0, 20.0],
        'mid_range': [20.0, 70.0],
        'overcast_range': [70.0, 100.0],
    },
    'precipitation': {
        'clear_max': 20.0,
        'mid_max': 60.0,
        'overcast_max': 85.0,
        'drizzle_range': [5.0, 20.0],
        'rain_range': [25.0, 55.0],
        'heavy_range': [60.0, 100.0],
        'probabilities': {
            'clear': {'none': 0.92, 'drizzle': 0.07, 'rain': 0.01, 'heavy': 0.0},
            'mid': {'none': 0.75, 'drizzle': 0.18, 'rain': 0.07, 'heavy': 0.0},
            'overcast': {'none': 0.45, 'drizzle': 0.30, 'rain': 0.20, 'heavy': 0.05},
            'storm': {'none': 0.20, 'drizzle': 0.25, 'rain': 0.35, 'heavy': 0.20},
        },
    },
    'wet_surface': {
        'high_cloudiness_threshold': 50.0,
        'recent_rain_probability_high_cloud': 0.12,
        'recent_rain_probability_low_cloud': 0.05,
        'dry_wetness_cloudiness_scale': 0.08,
        'dry_wetness_noise_std': 4.0,
        'dry_wetness_clip': [0.0, 18.0],
        'dry_deposits_range': [0.0, 5.0],
        'recent_rain_wetness_range': [20.0, 55.0],
        'recent_rain_deposits_range': [8.0, 28.0],
        'rain_wetness_precipitation_scale': 0.55,
        'rain_wetness_cloudiness_scale': 0.18,
        'rain_wetness_noise_std': 7.0,
        'rain_wetness_clip': [10.0, 100.0],
        'rain_deposits_precipitation_scale': 0.70,
        'rain_deposits_noise_std': 10.0,
        'rain_deposits_clip': [0.0, 100.0],
    },
    'wind': {
        'base': 5.0,
        'cloudiness_scale': 0.10,
        'precipitation_scale': 0.20,
        'noise_std': 8.0,
        'clip': [0.0, 65.0],
        'heavy_precipitation_threshold': 60.0,
        'heavy_precipitation_min': 20.0,
    },
    'fog': {
        'high_cloudiness_threshold': 50.0,
        'night_or_twilight_probability': 0.18,
        'day_probability': 0.08,
        'clear_scattering_intensity': 0.35,
        'clear_mie_scattering_scale': 0.05,
        'density_range': [5.0, 18.0],
        'distance_range': [35.0, 120.0],
        'distance_density_scale': 1.5,
        'distance_clip': [25.0, 120.0],
        'falloff_range': [0.4, 1.2],
        'scattering_range': [0.15, 0.45],
        'mie_range': [0.02, 0.08],
        'disabled_distance': 1000.0,
        'disabled_falloff': 1.0,
    },
    'postprocess': {
        'rayleigh_scattering_scale': 0.06,
        'dust_storm': 0.0,
    },
}

WEATHER_FIELD_NAMES = (
    'cloudiness',
    'precipitation',
    'precipitation_deposits',
    'wind_intensity',
    'sun_azimuth_angle',
    'sun_altitude_angle',
    'fog_density',
    'fog_distance',
    'wetness',
    'fog_falloff',
    'scattering_intensity',
    'mie_scattering_scale',
    'rayleigh_scattering_scale',
    'dust_storm',
)

LEGACY_SAMPLING_KEYS = (
    'sun_altitude',
    'sun_azimuth',
    'cloudiness',
    'precipitation',
    'wet_surface',
    'wind',
    'fog',
    'postprocess',
)


class WeatherGeneratorError(ValueError):
    pass


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _float_range(config, key):
    values = config[key]
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise WeatherGeneratorError(f'weather.{key} must be a two-value range.')
    low = float(values[0])
    high = float(values[1])
    if high < low:
        raise WeatherGeneratorError(f'weather.{key} must have low <= high.')
    return low, high


def _float_clip(config, key):
    return _float_range(config, key)


def _weighted_choice(rng, weights, config_name):
    normalized = []
    total = 0.0
    for name, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight < 0.0:
            raise WeatherGeneratorError(f'{config_name}.{name} weight must be >= 0.')
        normalized.append((name, weight))
        total += weight

    if total <= 0.0:
        raise WeatherGeneratorError(f'{config_name} must have a positive total weight.')

    threshold = rng.uniform(0.0, total)
    cumulative = 0.0
    for name, weight in normalized:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return normalized[-1][0]


def _uniform(rng, bounds):
    low, high = bounds
    return low if abs(high - low) <= 1e-9 else rng.uniform(low, high)


def _clamp(value, bounds):
    low, high = bounds
    return max(low, min(high, value))


def _normalize_01(value, bounds):
    low, high = bounds
    if abs(high - low) <= 1e-9:
        return 0.0
    return _clamp((value - low) / (high - low), (0.0, 1.0))


def _normal_clipped(rng, mean, std, bounds):
    if std <= 0.0:
        return _clamp(mean, bounds)
    return _clamp(rng.gauss(mean, std), bounds)


def _resolve_weather_config(weather_config, fallback_seed):
    resolved = copy.deepcopy(DEFAULT_WEATHER_CONFIG)
    legacy_sampling = False
    if weather_config:
        legacy_sampling = any(key in weather_config for key in LEGACY_SAMPLING_KEYS)
        deep_update(resolved, weather_config)

    if resolved.get('seed') is None:
        resolved['seed'] = int(fallback_seed if fallback_seed is not None else 0)
    else:
        resolved['seed'] = int(resolved['seed'])

    resolved['_legacy_sampling'] = legacy_sampling
    return resolved


def _validate_weather_config(config):
    if not isinstance(config.get('regimes'), dict):
        raise WeatherGeneratorError('weather.regimes must be a table.')
    _weighted_choice(random.Random(0), config['regimes']['weights'], 'weather.regimes.weights')
    for preset_name in config['regimes']['weights']:
        if preset_name not in OFFICIAL_PRESET_FIELDS:
            raise WeatherGeneratorError(f'weather.regimes.weights.{preset_name} is not a known CARLA weather preset.')

    if not isinstance(config.get('realism'), dict):
        raise WeatherGeneratorError('weather.realism must be a table.')
    max_attempts = int(config['realism'].get('max_resample_attempts', 1))
    if max_attempts < 1:
        raise WeatherGeneratorError('weather.realism.max_resample_attempts must be >= 1.')
    for key in ('min_scattering_intensity', 'min_mie_scattering_scale', 'min_rayleigh_scattering_scale'):
        if float(config['realism'].get(key, 0.0)) < 0.0:
            raise WeatherGeneratorError(f'weather.realism.{key} must be >= 0.')
    if not isinstance(config.get('preset_jitter'), dict):
        raise WeatherGeneratorError('weather.preset_jitter must be a table.')
    if float(config['preset_jitter'].get('scale', 1.0)) < 0.0:
        raise WeatherGeneratorError('weather.preset_jitter.scale must be >= 0.')

    legacy_sections = (
        ('sun_altitude', config['sun_altitude']),
        ('sun_azimuth', config['sun_azimuth']),
        ('cloudiness', config['cloudiness']),
        ('precipitation', config['precipitation']),
        ('wet_surface', config['wet_surface']),
        ('wind', config['wind']),
        ('fog', config['fog']),
    )
    for section_name, section in legacy_sections:
        if not isinstance(section, dict):
            raise WeatherGeneratorError(f'weather.{section_name} must be a table.')

    for key in ('day_range', 'low_sun_range', 'twilight_range', 'night_range'):
        _float_range(config['sun_altitude'], key)
    _float_range(config['sun_azimuth'], 'range')
    for key in ('clear_range', 'mid_range', 'overcast_range'):
        _float_range(config['cloudiness'], key)
    for key in ('drizzle_range', 'rain_range', 'heavy_range'):
        _float_range(config['precipitation'], key)
    for key in ('dry_wetness_clip', 'dry_deposits_range', 'recent_rain_wetness_range',
                'recent_rain_deposits_range', 'rain_wetness_clip', 'rain_deposits_clip'):
        _float_range(config['wet_surface'], key)
    _float_range(config['wind'], 'clip')
    for key in ('density_range', 'distance_range', 'distance_clip', 'falloff_range',
                'scattering_range', 'mie_range'):
        _float_range(config['fog'], key)

    _weighted_choice(random.Random(0), config['sun_altitude']['weights'], 'weather.sun_altitude.weights')
    _weighted_choice(random.Random(0), config['cloudiness']['weights'], 'weather.cloudiness.weights')
    for bucket_name, bucket_weights in config['precipitation']['probabilities'].items():
        _weighted_choice(
            random.Random(0),
            bucket_weights,
            f'weather.precipitation.probabilities.{bucket_name}',
        )


def _cloudiness_bucket(cloudiness, precipitation_config):
    if cloudiness < float(precipitation_config['clear_max']):
        return 'clear'
    if cloudiness < float(precipitation_config['mid_max']):
        return 'mid'
    if cloudiness < float(precipitation_config['overcast_max']):
        return 'overcast'
    return 'storm'


def _sample_precipitation(rng, config, cloudiness):
    bucket = _cloudiness_bucket(cloudiness, config)
    regime = _weighted_choice(
        rng,
        config['probabilities'][bucket],
        f'weather.precipitation.probabilities.{bucket}',
    )

    if regime == 'none':
        value = 0.0
    else:
        value = _uniform(rng, _float_range(config, f'{regime}_range'))

    return bucket, regime, value


def _sample_fog(rng, config, sun_bucket, cloudiness):
    fog_probability = float(config['day_probability'])
    if sun_bucket in ('twilight', 'night') and cloudiness >= float(config['high_cloudiness_threshold']):
        fog_probability = float(config['night_or_twilight_probability'])

    enabled = rng.random() < fog_probability
    if not enabled:
        return {
            'enabled': False,
            'fog_density': 0.0,
            'fog_distance': float(config['disabled_distance']),
            'fog_falloff': float(config['disabled_falloff']),
            'scattering_intensity': float(config.get('clear_scattering_intensity', 0.2)),
            'mie_scattering_scale': float(config.get('clear_mie_scattering_scale', 0.03)),
        }

    fog_density = _uniform(rng, _float_range(config, 'density_range'))
    raw_distance = _uniform(rng, _float_range(config, 'distance_range'))
    raw_distance -= float(config['distance_density_scale']) * fog_density
    fog_distance = _clamp(raw_distance, _float_clip(config, 'distance_clip'))

    return {
        'enabled': True,
        'fog_density': fog_density,
        'fog_distance': fog_distance,
        'fog_falloff': _uniform(rng, _float_range(config, 'falloff_range')),
        'scattering_intensity': _uniform(rng, _float_range(config, 'scattering_range')),
        'mie_scattering_scale': _uniform(rng, _float_range(config, 'mie_range')),
    }


def _sample_regime(rng, config):
    return _weighted_choice(rng, config['regimes']['weights'], 'weather.regimes.weights')


def _preset_category(preset_name):
    if preset_name.endswith('Night'):
        return 'night'
    if preset_name.endswith('Sunset'):
        return 'sunset'
    return 'day'


def _jitter_positive_field(rng, fields, config, field_name, std_key):
    jitter_cfg = config['preset_jitter']
    if not jitter_cfg.get('enabled', True):
        return
    base_value = fields[field_name]
    if base_value < 0.0:
        return
    scale = float(jitter_cfg.get('scale', 1.0))
    std = float(jitter_cfg.get(std_key, 0.0)) * scale
    fields[field_name] = _normal_clipped(rng, base_value, std, (0.0, 100.0))


def _jitter_sun_altitude(rng, fields, config, category):
    jitter_cfg = config['preset_jitter']
    if not jitter_cfg.get('enabled', True):
        return
    base_value = fields['sun_altitude_angle']
    scale = float(jitter_cfg.get('scale', 1.0))
    std = float(jitter_cfg.get('sun_altitude_std', 0.0)) * scale
    sun_cfg = config['sun_altitude']
    if category == 'day':
        bounds = _float_range(sun_cfg, 'day_range')
    elif category == 'sunset':
        bounds = _float_range(sun_cfg, 'low_sun_range')
    else:
        bounds = _float_range(sun_cfg, 'night_range')
        if base_value < bounds[0]:
            fields['sun_altitude_angle'] = _uniform(rng, bounds)
            return
    fields['sun_altitude_angle'] = _normal_clipped(rng, base_value, std, bounds)


def _sample_preset_weather_fields(config):
    rng = random.Random(int(config['seed']))
    preset_name = _sample_regime(rng, config)
    category = _preset_category(preset_name)
    fields = copy.deepcopy(OFFICIAL_PRESET_FIELDS[preset_name])

    for field_name, std_key in (
        ('cloudiness', 'cloudiness_std'),
        ('precipitation', 'precipitation_std'),
        ('precipitation_deposits', 'precipitation_deposits_std'),
        ('wind_intensity', 'wind_intensity_std'),
        ('wetness', 'wetness_std'),
    ):
        _jitter_positive_field(rng, fields, config, field_name, std_key)
    _jitter_sun_altitude(rng, fields, config, category)

    metadata = {
        'seed': int(config['seed']),
        'regime': preset_name,
        'regime_category': category,
        'preset': preset_name,
        'sun_bucket': _sun_bucket_from_altitude(fields['sun_altitude_angle']),
        'cloudiness_bucket': _cloud_bucket_from_cloudiness(fields['cloudiness']),
        'precipitation_bucket': _cloudiness_bucket(fields['cloudiness'], config['precipitation']),
        'precipitation_regime': _precipitation_regime_from_value(fields['precipitation']),
        'recent_rain': fields['precipitation'] <= 1e-6 and fields['precipitation_deposits'] > 1e-6,
        'fog_enabled': fields['fog_density'] > 1e-6,
        'parameters': {name: round(float(value), 4) for name, value in fields.items()},
        'generator_config': _public_generator_config(config),
        'sampling_mode': 'official_preset',
    }
    return fields, metadata


def _sun_bucket_from_altitude(sun_altitude):
    if sun_altitude >= 25.0:
        return 'day'
    if sun_altitude >= 8.0:
        return 'low_sun'
    if sun_altitude >= -8.0:
        return 'twilight'
    return 'night'


def _cloud_bucket_from_cloudiness(cloudiness):
    if cloudiness < 20.0:
        return 'clear'
    if cloudiness < 70.0:
        return 'mid'
    return 'overcast'


def _precipitation_regime_from_value(precipitation):
    if precipitation <= 1e-6:
        return 'none'
    if precipitation < 20.0:
        return 'drizzle'
    if precipitation < 55.0:
        return 'rain'
    return 'heavy'


def _sample_regime_precipitation(rng, regime):
    probability = float(regime.get('precipitation_probability', 0.0))
    if rng.random() >= probability:
        return 0.0
    return _uniform(rng, _float_range(regime, 'precipitation_range'))


def _sample_regime_fog(rng, regime, precipitation, wind_intensity):
    enabled = rng.random() < float(regime.get('fog_probability', 0.0))
    if precipitation > 15.0 or wind_intensity > 18.0:
        enabled = False

    if not enabled:
        return {
            'enabled': False,
            'fog_density': 0.0,
            'fog_distance': 1000.0,
            'fog_falloff': 1.0,
        }

    fog_density = _uniform(rng, _float_range(regime, 'fog_density_range'))
    fog_distance = _uniform(rng, _float_range(regime, 'fog_distance_range'))
    fog_distance -= 1.2 * fog_density
    fog_distance = _clamp(fog_distance, (35.0, 1000.0))

    return {
        'enabled': True,
        'fog_density': fog_density,
        'fog_distance': fog_distance,
        'fog_falloff': _uniform(rng, (0.4, 1.2)),
    }


def _derive_regime_surface(rng, regime, cloudiness, precipitation):
    if precipitation > 1e-6:
        wetness_mean = 0.55 * precipitation + 0.12 * cloudiness
        wetness = _normal_clipped(rng, wetness_mean, 6.0, (15.0, 90.0))
        deposits_mean = 0.55 * precipitation
        precipitation_deposits = _normal_clipped(rng, deposits_mean, 8.0, (0.0, 70.0))
        return wetness, precipitation_deposits, False

    recent_rain = rng.random() < float(regime.get('post_rain_probability', 0.0))
    if recent_rain:
        wetness = _uniform(rng, (20.0, 55.0))
        precipitation_deposits = _uniform(rng, (6.0, 24.0))
        return wetness, precipitation_deposits, True

    wetness_mean = 0.10 * cloudiness
    wetness = _normal_clipped(rng, wetness_mean, 3.0, (0.0, 18.0))
    precipitation_deposits = _uniform(rng, (0.0, 4.0))
    return wetness, precipitation_deposits, False


def _brightness_score(fields, category):
    sun = fields['sun_altitude_angle']
    if category == 'night':
        sun_score = _normalize_01(sun, (-12.0, -2.0)) * 0.35
        base = 0.08
    elif category == 'sunset':
        sun_score = _normalize_01(sun, (-3.0, 18.0)) * 0.55
        base = 0.15
    else:
        sun_score = _normalize_01(sun, (10.0, 75.0)) * 0.75
        base = 0.20

    cloud_penalty = 0.32 * (fields['cloudiness'] / 100.0)
    rain_penalty = 0.18 * (fields['precipitation'] / 100.0)
    fog_penalty = 0.22 * (fields['fog_density'] / 30.0)
    scattering_bonus = 0.18 * _normalize_01(fields['scattering_intensity'], (0.25, 0.72))
    mie_bonus = 0.07 * _normalize_01(fields['mie_scattering_scale'], (0.03, 0.09))
    return _clamp(
        base + sun_score + scattering_bonus + mie_bonus - cloud_penalty - rain_penalty - fog_penalty,
        (0.0, 1.0),
    )


def _is_regime_candidate_realistic(fields, regime, brightness):
    category = regime['category']
    sun = fields['sun_altitude_angle']
    cloudiness = fields['cloudiness']
    precipitation = fields['precipitation']
    fog_density = fields['fog_density']
    wind_intensity = fields['wind_intensity']

    if category != 'night' and sun < -3.5:
        return False
    if category == 'night' and sun < -12.5:
        return False
    if category != 'night' and sun < 12.0 and cloudiness > 80.0 and precipitation > 20.0:
        return False
    if fog_density > 18.0 and (wind_intensity >= 10.0 or precipitation >= 10.0):
        return False
    if precipitation > 35.0 and cloudiness < 90.0:
        return False
    if cloudiness < 20.0 and (precipitation > 1e-6 or fields['wetness'] > 18.0):
        return False
    return brightness >= float(regime['min_brightness'])


def _sample_regime_candidate(rng, config, regime_name):
    regime = REGIME_DEFINITIONS[regime_name]
    realism_cfg = config['realism']

    sun_altitude = _uniform(rng, _float_range(regime, 'sun_altitude_range'))
    sun_azimuth = _uniform(rng, (0.0, 360.0))
    cloudiness = _uniform(rng, _float_range(regime, 'cloudiness_range'))

    precipitation = _sample_regime_precipitation(rng, regime)
    if precipitation > 15.0:
        cloudiness = max(cloudiness, 70.0)
    if precipitation > 35.0:
        cloudiness = max(cloudiness, 90.0)
    if cloudiness < 20.0:
        precipitation = 0.0

    wind_intensity = _uniform(rng, _float_range(regime, 'wind_range'))
    fog = _sample_regime_fog(rng, regime, precipitation, wind_intensity)
    wetness, precipitation_deposits, recent_rain = _derive_regime_surface(
        rng,
        regime,
        cloudiness,
        precipitation,
    )

    scattering_intensity = max(
        _uniform(rng, _float_range(regime, 'scattering_range')),
        float(realism_cfg['min_scattering_intensity']),
    )
    mie_scattering_scale = max(
        _uniform(rng, _float_range(regime, 'mie_range')),
        float(realism_cfg['min_mie_scattering_scale']),
    )
    rayleigh_scattering_scale = max(
        _uniform(rng, _float_range(regime, 'rayleigh_range')),
        float(realism_cfg['min_rayleigh_scattering_scale']),
    )

    weather_fields = {
        'cloudiness': cloudiness,
        'precipitation': precipitation,
        'precipitation_deposits': precipitation_deposits,
        'wind_intensity': wind_intensity,
        'sun_azimuth_angle': sun_azimuth,
        'sun_altitude_angle': sun_altitude,
        'fog_density': fog['fog_density'],
        'fog_distance': fog['fog_distance'],
        'wetness': wetness,
        'fog_falloff': fog['fog_falloff'],
        'scattering_intensity': scattering_intensity,
        'mie_scattering_scale': mie_scattering_scale,
        'rayleigh_scattering_scale': rayleigh_scattering_scale,
        'dust_storm': 0.0,
    }
    brightness = _brightness_score(weather_fields, regime['category'])
    metadata = {
        'regime': regime_name,
        'regime_category': regime['category'],
        'sun_bucket': _sun_bucket_from_altitude(sun_altitude),
        'cloudiness_bucket': _cloud_bucket_from_cloudiness(cloudiness),
        'precipitation_bucket': _cloudiness_bucket(cloudiness, config['precipitation']),
        'precipitation_regime': _precipitation_regime_from_value(precipitation),
        'recent_rain': recent_rain,
        'fog_enabled': fog['enabled'],
        'brightness_score': round(float(brightness), 4),
        'realistic': _is_regime_candidate_realistic(weather_fields, regime, brightness),
    }
    return weather_fields, metadata


def _sample_regime_weather_fields(config):
    rng = random.Random(int(config['seed']))
    max_attempts = int(config['realism'].get('max_resample_attempts', 1))
    best_fields = None
    best_metadata = None
    regime_name = _sample_regime(rng, config)

    for attempt in range(1, max_attempts + 1):
        fields, metadata = _sample_regime_candidate(rng, config, regime_name)
        metadata['resample_attempts'] = attempt
        if best_metadata is None or metadata['brightness_score'] > best_metadata['brightness_score']:
            best_fields = fields
            best_metadata = metadata
        if metadata['realistic']:
            best_fields = fields
            best_metadata = metadata
            break

    best_metadata['seed'] = int(config['seed'])
    best_metadata['parameters'] = {name: round(float(value), 4) for name, value in best_fields.items()}
    best_metadata['generator_config'] = _public_generator_config(config)
    best_metadata['sampling_mode'] = 'regime'
    return best_fields, best_metadata


def _public_generator_config(config):
    public_config = copy.deepcopy(config)
    public_config.pop('_legacy_sampling', None)
    return public_config


def _sample_weather_fields(config):
    rng = random.Random(int(config['seed']))

    sun_bucket = _weighted_choice(rng, config['sun_altitude']['weights'], 'weather.sun_altitude.weights')
    sun_altitude = _uniform(rng, _float_range(config['sun_altitude'], f'{sun_bucket}_range'))
    sun_azimuth = _uniform(rng, _float_range(config['sun_azimuth'], 'range'))

    cloud_bucket = _weighted_choice(rng, config['cloudiness']['weights'], 'weather.cloudiness.weights')
    cloudiness = _uniform(rng, _float_range(config['cloudiness'], f'{cloud_bucket}_range'))

    precipitation_bucket, precipitation_regime, precipitation = _sample_precipitation(
        rng,
        config['precipitation'],
        cloudiness,
    )

    wet_cfg = config['wet_surface']
    if precipitation <= 1e-6:
        if cloudiness >= float(wet_cfg['high_cloudiness_threshold']):
            recent_rain_probability = float(wet_cfg['recent_rain_probability_high_cloud'])
        else:
            recent_rain_probability = float(wet_cfg['recent_rain_probability_low_cloud'])
        recent_rain = rng.random() < recent_rain_probability

        if recent_rain:
            wetness = _uniform(rng, _float_range(wet_cfg, 'recent_rain_wetness_range'))
            precipitation_deposits = _uniform(rng, _float_range(wet_cfg, 'recent_rain_deposits_range'))
        else:
            wetness_mean = float(wet_cfg['dry_wetness_cloudiness_scale']) * cloudiness
            wetness = _normal_clipped(
                rng,
                wetness_mean,
                float(wet_cfg['dry_wetness_noise_std']),
                _float_clip(wet_cfg, 'dry_wetness_clip'),
            )
            precipitation_deposits = _uniform(rng, _float_range(wet_cfg, 'dry_deposits_range'))
    else:
        recent_rain = False
        wetness_mean = (
            float(wet_cfg['rain_wetness_precipitation_scale']) * precipitation
            + float(wet_cfg['rain_wetness_cloudiness_scale']) * cloudiness
        )
        wetness = _normal_clipped(
            rng,
            wetness_mean,
            float(wet_cfg['rain_wetness_noise_std']),
            _float_clip(wet_cfg, 'rain_wetness_clip'),
        )
        deposits_mean = float(wet_cfg['rain_deposits_precipitation_scale']) * precipitation
        precipitation_deposits = _normal_clipped(
            rng,
            deposits_mean,
            float(wet_cfg['rain_deposits_noise_std']),
            _float_clip(wet_cfg, 'rain_deposits_clip'),
        )

    wind_cfg = config['wind']
    wind_mean = (
        float(wind_cfg['base'])
        + float(wind_cfg['cloudiness_scale']) * cloudiness
        + float(wind_cfg['precipitation_scale']) * precipitation
    )
    wind_intensity = _normal_clipped(
        rng,
        wind_mean,
        float(wind_cfg['noise_std']),
        _float_clip(wind_cfg, 'clip'),
    )
    if precipitation >= float(wind_cfg['heavy_precipitation_threshold']):
        wind_intensity = max(wind_intensity, float(wind_cfg['heavy_precipitation_min']))

    fog = _sample_fog(rng, config['fog'], sun_bucket, cloudiness)
    postprocess_cfg = config['postprocess']

    weather_fields = {
        'cloudiness': cloudiness,
        'precipitation': precipitation,
        'precipitation_deposits': precipitation_deposits,
        'wind_intensity': wind_intensity,
        'sun_azimuth_angle': sun_azimuth,
        'sun_altitude_angle': sun_altitude,
        'fog_density': fog['fog_density'],
        'fog_distance': fog['fog_distance'],
        'wetness': wetness,
        'fog_falloff': fog['fog_falloff'],
        'scattering_intensity': fog['scattering_intensity'],
        'mie_scattering_scale': fog['mie_scattering_scale'],
        'rayleigh_scattering_scale': float(postprocess_cfg['rayleigh_scattering_scale']),
        'dust_storm': float(postprocess_cfg['dust_storm']),
    }

    metadata = {
        'seed': int(config['seed']),
        'sun_bucket': sun_bucket,
        'cloudiness_bucket': cloud_bucket,
        'precipitation_bucket': precipitation_bucket,
        'precipitation_regime': precipitation_regime,
        'recent_rain': recent_rain,
        'fog_enabled': fog['enabled'],
        'parameters': {name: round(float(value), 4) for name, value in weather_fields.items()},
        'generator_config': _public_generator_config(config),
        'sampling_mode': 'legacy_bucket',
    }
    return weather_fields, metadata


def generate_weather(weather_config, fallback_seed=None):
    resolved_config = _resolve_weather_config(weather_config, fallback_seed)
    if not resolved_config.get('enabled', True):
        return None, None

    _validate_weather_config(resolved_config)
    if resolved_config.get('_legacy_sampling'):
        fields, metadata = _sample_weather_fields(resolved_config)
    else:
        fields, metadata = _sample_preset_weather_fields(resolved_config)

    weather = carla.WeatherParameters()
    for field_name in WEATHER_FIELD_NAMES:
        if field_name in fields:
            setattr(weather, field_name, float(fields[field_name]))

    logging.info(
        'Generated weather seed=%d mode=%s regime=%s sun=%s cloud=%s rain=%s fog=%s params=%s',
        metadata['seed'],
        metadata.get('sampling_mode', 'regime'),
        metadata.get('regime', 'n/a'),
        metadata['sun_bucket'],
        metadata['cloudiness_bucket'],
        metadata['precipitation_regime'],
        metadata['fog_enabled'],
        json.dumps(metadata['parameters'], sort_keys=True),
    )
    return weather, metadata
