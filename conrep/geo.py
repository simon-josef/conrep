"""
geo.py
------
Geographic and sociodemographic predictor construction for Avenue 2.

The single public entry point is load_swow_predictors(), which handles
everything: GeoNames download, coordinate lookup, predictor computation,
and World Bank API calls. The notebook only calls this one function.

If the input DataFrame lacks city/country/age columns, the function
warns and returns whatever predictors it can compute.

Available predictors
--------------------
age                     absolute age (years) — requires 'age' column
abs_latitude            absolute latitude of participant city
geographic_distance     great-circle distance between participant cities (km)
city_population         population of participant city (GeoNames)
distance_to_capital     great-circle distance to country capital (km)
country_population      total country population (World Bank API)
country_gdp_per_capita  GDP per capita in USD (World Bank API)
"""

import math
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import pycountry


GEONAMES_URL  = "https://download.geonames.org/export/dump/cities500.zip"
GEONAMES_ZIP  = "cities500.zip"
GEONAMES_FILE = "cities500.txt"
GEONAMES_COLS = [
    'geonameid', 'name', 'asciiname', 'alternatenames', 'lat', 'lon',
    'feature_class', 'feature_code', 'country_code', 'cc2',
    'admin1', 'admin2', 'admin3', 'admin4', 'population',
    'elevation', 'dem', 'timezone', 'modified'
]

COUNTRY_CAPITALS = {
    'AF': 'Kabul', 'AL': 'Tirana', 'DZ': 'Algiers', 'AD': 'Andorra la Vella',
    'AO': 'Luanda', 'AG': 'Saint Johns', 'AR': 'Buenos Aires', 'AM': 'Yerevan',
    'AU': 'Canberra', 'AT': 'Vienna', 'AZ': 'Baku', 'BS': 'Nassau',
    'BH': 'Manama', 'BD': 'Dhaka', 'BB': 'Bridgetown', 'BY': 'Minsk',
    'BE': 'Brussels', 'BZ': 'Belmopan', 'BJ': 'Porto-Novo', 'BT': 'Thimphu',
    'BO': 'Sucre', 'BA': 'Sarajevo', 'BW': 'Gaborone', 'BR': 'Brasilia',
    'BN': 'Bandar Seri Begawan', 'BG': 'Sofia', 'BF': 'Ouagadougou',
    'BI': 'Gitega', 'CV': 'Praia', 'KH': 'Phnom Penh', 'CM': 'Yaounde',
    'CA': 'Ottawa', 'CF': 'Bangui', 'TD': "N'Djamena", 'CL': 'Santiago',
    'CN': 'Beijing', 'CO': 'Bogota', 'KM': 'Moroni', 'CD': 'Kinshasa',
    'CG': 'Brazzaville', 'CR': 'San Jose', 'HR': 'Zagreb', 'CU': 'Havana',
    'CY': 'Nicosia', 'CZ': 'Prague', 'DK': 'Copenhagen', 'DJ': 'Djibouti',
    'DM': 'Roseau', 'DO': 'Santo Domingo', 'EC': 'Quito', 'EG': 'Cairo',
    'SV': 'San Salvador', 'GQ': 'Malabo', 'ER': 'Asmara', 'EE': 'Tallinn',
    'SZ': 'Mbabane', 'ET': 'Addis Ababa', 'FJ': 'Suva', 'FI': 'Helsinki',
    'FR': 'Paris', 'GA': 'Libreville', 'GM': 'Banjul', 'GE': 'Tbilisi',
    'DE': 'Berlin', 'GH': 'Accra', 'GR': 'Athens', 'GD': "Saint George's",
    'GT': 'Guatemala City', 'GN': 'Conakry', 'GW': 'Bissau', 'GY': 'Georgetown',
    'HT': 'Port-au-Prince', 'HN': 'Tegucigalpa', 'HU': 'Budapest',
    'IS': 'Reykjavik', 'IN': 'New Delhi', 'ID': 'Jakarta', 'IR': 'Tehran',
    'IQ': 'Baghdad', 'IE': 'Dublin', 'IL': 'Jerusalem', 'IT': 'Rome',
    'JM': 'Kingston', 'JP': 'Tokyo', 'JO': 'Amman', 'KZ': 'Astana',
    'KE': 'Nairobi', 'KI': 'South Tarawa', 'KP': 'Pyongyang', 'KR': 'Seoul',
    'KW': 'Kuwait City', 'KG': 'Bishkek', 'LA': 'Vientiane', 'LV': 'Riga',
    'LB': 'Beirut', 'LS': 'Maseru', 'LR': 'Monrovia', 'LY': 'Tripoli',
    'LI': 'Vaduz', 'LT': 'Vilnius', 'LU': 'Luxembourg', 'MG': 'Antananarivo',
    'MW': 'Lilongwe', 'MY': 'Kuala Lumpur', 'MV': 'Male', 'ML': 'Bamako',
    'MT': 'Valletta', 'MH': 'Majuro', 'MR': 'Nouakchott', 'MU': 'Port Louis',
    'MX': 'Mexico City', 'FM': 'Palikir', 'MD': 'Chisinau', 'MC': 'Monaco',
    'MN': 'Ulaanbaatar', 'ME': 'Podgorica', 'MA': 'Rabat', 'MZ': 'Maputo',
    'MM': 'Naypyidaw', 'NA': 'Windhoek', 'NR': 'Yaren', 'NP': 'Kathmandu',
    'NL': 'Amsterdam', 'NZ': 'Wellington', 'NI': 'Managua', 'NE': 'Niamey',
    'NG': 'Abuja', 'MK': 'Skopje', 'NO': 'Oslo', 'OM': 'Muscat',
    'PK': 'Islamabad', 'PW': 'Ngerulmud', 'PA': 'Panama City', 'PG': 'Port Moresby',
    'PY': 'Asuncion', 'PE': 'Lima', 'PH': 'Manila', 'PL': 'Warsaw',
    'PT': 'Lisbon', 'QA': 'Doha', 'RO': 'Bucharest', 'RU': 'Moscow',
    'RW': 'Kigali', 'KN': 'Basseterre', 'LC': 'Castries', 'VC': 'Kingstown',
    'WS': 'Apia', 'SM': 'San Marino', 'ST': 'Sao Tome', 'SA': 'Riyadh',
    'SN': 'Dakar', 'RS': 'Belgrade', 'SC': 'Victoria', 'SL': 'Freetown',
    'SG': 'Singapore', 'SK': 'Bratislava', 'SI': 'Ljubljana', 'SB': 'Honiara',
    'SO': 'Mogadishu', 'ZA': 'Pretoria', 'SS': 'Juba', 'ES': 'Madrid',
    'LK': 'Sri Jayawardenepura Kotte', 'SD': 'Khartoum', 'SR': 'Paramaribo',
    'SE': 'Stockholm', 'CH': 'Bern', 'SY': 'Damascus', 'TW': 'Taipei',
    'TJ': 'Dushanbe', 'TZ': 'Dodoma', 'TH': 'Bangkok', 'TL': 'Dili',
    'TG': 'Lome', 'TO': "Nuku'alofa", 'TT': 'Port of Spain', 'TN': 'Tunis',
    'TR': 'Ankara', 'TM': 'Ashgabat', 'TV': 'Funafuti', 'UG': 'Kampala',
    'UA': 'Kiev', 'AE': 'Abu Dhabi', 'GB': 'London', 'US': 'Washington D.C.',
    'UY': 'Montevideo', 'UZ': 'Tashkent', 'VU': 'Port Vila', 'VE': 'Caracas',
    'VN': 'Hanoi', 'YE': 'Sanaa', 'ZM': 'Lusaka', 'ZW': 'Harare',
}


class GeoCoordinates:
    """A lazy participant-distance object for geographic distance.

    Stores each participant's (lat, lon) and exposes an interface compatible
    with build_dyads(): an .index attribute and value lookup via __getitem__
    that returns the coordinate pair. build_dyads() detects this type and
    computes the great-circle distance between two participants instead of
    a plain numeric difference.

    This avoids precomputing an O(n^2) distance matrix over the full
    participant pool — the distance is only ever computed for the small
    number of participant pairs that responded to a given concept.
    """

    def __init__(self, lat: pd.Series, lon: pd.Series):
        self.lat = lat
        self.lon = lon
        self.index = lat.index

    def __getitem__(self, pid):
        return (self.lat[pid], self.lon[pid])

    @staticmethod
    def distance(coord_i, coord_j) -> float:
        """Great-circle distance (km) between two (lat, lon) pairs."""
        lat_i, lon_i = coord_i
        lat_j, lon_j = coord_j
        if pd.isna(lat_i) or pd.isna(lat_j) or pd.isna(lon_i) or pd.isna(lon_j):
            return np.nan
        return _haversine(lat_i, lon_i, lat_j, lon_j)


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two points (degrees). Public wrapper."""
    return _haversine(lat1, lon1, lat2, lon2)


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R    = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _country_to_iso2(name: str):
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        return np.nan


def _load_geonames() -> pd.DataFrame:
    if not os.path.exists(GEONAMES_FILE):
        print("Downloading GeoNames cities500 (~30 MB)...")
        urllib.request.urlretrieve(GEONAMES_URL, GEONAMES_ZIP)
        with zipfile.ZipFile(GEONAMES_ZIP) as z:
            z.extract(GEONAMES_FILE)
        print("Download complete.")
    return pd.read_csv(
        GEONAMES_FILE, sep='\t', header=None,
        names=GEONAMES_COLS, low_memory=False
    )


def load_swow_predictors(
    df: pd.DataFrame,
    participant_col: str = 'participantID',
) -> dict:
    """Compute all available participant-level predictors from a SWOW-format DataFrame.

    Downloads GeoNames automatically if not already present. Fetches country
    population and GDP per capita from the World Bank API (requires internet).

    Predictors are computed only for columns that exist in df. Missing columns
    are skipped with a warning rather than raising an error, so the function
    works with any subset of SWOW metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered association DataFrame. Relevant columns (all optional):
        'age', 'city', 'country'.
    participant_col : str

    Returns
    -------
    dict
        Keys are predictor names (str). Values are pd.Series indexed by
        participantID. Pass any entry directly to build_dyads().
    """
    predictors = {}
    required_geo = {'city', 'country'}
    has_geo = required_geo.issubset(df.columns)
    has_age = 'age' in df.columns

    df_p = (
        df[[participant_col] + [c for c in ['city', 'country', 'age'] if c in df.columns]]
        .drop_duplicates(subset=participant_col)
        .copy()
    )

    # Age
    if has_age:
        predictors['age'] = (
            df_p.set_index(participant_col)['age']
            .apply(pd.to_numeric, errors='coerce')
        )
    else:
        print("Warning: 'age' column not found — skipping age predictor.")

    # Geographic predictors
    if has_geo:
        df_geo = _load_geonames()
        df_geo['city_norm'] = df_geo['asciiname'].str.strip().str.lower()

        for col in ['city', 'country']:
            df_p[col] = (
                df_p[col].astype(str).str.strip()
                .replace({'': np.nan, 'nan': np.nan, 'unknown': np.nan})
            )

        df_p['country_code'] = df_p['country'].apply(_country_to_iso2)
        df_p['city_norm']    = df_p['city'].str.strip().str.lower()

        df_geo_dedup = (
            df_geo.sort_values('population', ascending=False)
            .drop_duplicates(subset=['city_norm', 'country_code'])
            [['city_norm', 'country_code', 'lat', 'lon', 'population']]
        )

        df_p = (
            df_p.drop(columns=[c for c in ['lat', 'lon'] if c in df_p.columns])
            .merge(df_geo_dedup, on=['city_norm', 'country_code'], how='left')
            .rename(columns={'population': 'city_population'})
        )
        df_p['abs_latitude'] = df_p['lat'].abs()

        # Capital coordinates from GeoNames
        cap_coords = {}
        for iso2, capital in COUNTRY_CAPITALS.items():
            cap_norm = capital.strip().lower()
            match = df_geo[
                (df_geo['city_norm'] == cap_norm) &
                (df_geo['country_code'] == iso2)
            ]
            if len(match) == 0:
                match = df_geo[df_geo['city_norm'] == cap_norm]
            if len(match) > 0:
                row = match.iloc[0]
                cap_coords[iso2] = (row['lat'], row['lon'])

        df_p['capital_lat'] = df_p['country_code'].map(
            lambda x: cap_coords.get(x, (np.nan, np.nan))[0]
        )
        df_p['capital_lon'] = df_p['country_code'].map(
            lambda x: cap_coords.get(x, (np.nan, np.nan))[1]
        )

        df_p_valid = df_p.dropna(subset=['lat', 'lon']).copy()

        predictors['abs_latitude']   = df_p_valid.set_index(participant_col)['abs_latitude']
        predictors['city_population']     = df_p_valid.set_index(participant_col)['city_population']
        predictors['city_population_log'] = np.log1p(predictors['city_population'])

        dist_cap = df_p_valid.apply(
            lambda r: _haversine(r['lat'], r['lon'], r['capital_lat'], r['capital_lon'])
            if pd.notna(r['lat']) and pd.notna(r['capital_lat']) else np.nan,
            axis=1
        )
        predictors['distance_to_capital'] = pd.Series(
            dist_cap.values, index=df_p_valid[participant_col]
        )

        # Geographic distance — a lazy coordinate-pair object, not a precomputed
        # global distance matrix. Computing distance for all participant pairs
        # up front would be O(n^2) and far too slow for SWOW-scale data (tens
        # of thousands of participants). Instead, GeoCoordinates stores each
        # participant's (lat, lon) and computes the great-circle distance
        # on demand, only for the participant pairs build_dyads() actually
        # needs — which is the much smaller set who responded to a given
        # concept (typically tens to a few hundred).
        coords = df_p_valid.set_index(participant_col)[['lat', 'lon']]
        predictors['geographic_distance'] = GeoCoordinates(coords['lat'], coords['lon'])

        # World Bank — population and GDP per capita.
        # Uses wbgapi rather than world_bank_data: wbgapi is actively
        # maintained, handles large country lists internally without the
        # URL-length failures that plain semicolon-joined requests can hit,
        # and is the package the World Bank itself points to. It is
        # installed automatically if missing (same philosophy as the
        # GeoNames download above). If the API call itself fails (e.g.
        # unreachable or rate-limited), this block fails gracefully and
        # skips these two predictors rather than blocking the rest of the
        # function.
        country_codes = df_p_valid['country_code'].dropna().unique().tolist()
        try:
            try:
                import wbgapi as wb
            except ImportError:
                print("wbgapi not found — installing automatically...")
                import subprocess, sys
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--quiet", "wbgapi"
                    ])
                except subprocess.CalledProcessError:
                    # Some environments (e.g. system-managed Python installs)
                    # refuse pip installs without this flag.
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--quiet",
                        "--break-system-packages", "wbgapi"
                    ])
                import wbgapi as wb

            iso3_map = {}
            for code in country_codes:
                try:
                    iso3_map[code] = pycountry.countries.get(alpha_2=code).alpha_3
                except AttributeError:
                    pass
            iso3_codes   = list(iso3_map.values())
            iso3_to_iso2 = {v: k for k, v in iso3_map.items()}

            print(f"Fetching World Bank data for {len(iso3_codes)} countries...")

            df_pop = wb.data.DataFrame('SP.POP.TOTL',    iso3_codes, mrv=1).iloc[:, 0]
            df_gdp = wb.data.DataFrame('NY.GDP.PCAP.CD', iso3_codes, mrv=1).iloc[:, 0]

            pop_map = {iso3_to_iso2.get(k, k): v for k, v in df_pop.items()}
            gdp_map = {iso3_to_iso2.get(k, k): v for k, v in df_gdp.items()}
            df_p_valid = df_p_valid.copy()
            df_p_valid['country_population']     = df_p_valid['country_code'].map(pop_map)
            df_p_valid['country_gdp_per_capita'] = df_p_valid['country_code'].map(gdp_map)
            predictors['country_population']         = df_p_valid.set_index(participant_col)['country_population']
            predictors['country_population_log']     = np.log1p(predictors['country_population'])
            predictors['country_gdp_per_capita']     = df_p_valid.set_index(participant_col)['country_gdp_per_capita']
            predictors['country_gdp_per_capita_log'] = np.log1p(predictors['country_gdp_per_capita'])
            print("World Bank data retrieved.")
        except ImportError as e:
            print(f"Warning: automatic installation of wbgapi failed ({e}) — "
                  f"skipping country_population and country_gdp_per_capita. "
                  f"Try installing manually: pip install wbgapi")
        except Exception as e:
            print(f"Warning: World Bank API call failed ({e}) — skipping "
                  f"country_population and country_gdp_per_capita.")

        print(f"Participants with valid coordinates: {len(df_p_valid)}")

        # Attach the participant-level geo table itself, not just individual
        # predictors. build_swow_subgroups() needs the full table (with
        # city_population and country_code columns) to build urban/rural and
        # continent subgroups. Stored under a private key, not printed in the
        # predictor summary above.
        predictors['_df_participants'] = df_p_valid
    else:
        print("Warning: 'city' and/or 'country' columns not found — skipping geographic predictors.")

    print("\nAvailable predictors:")
    for name, s in predictors.items():
        if name.startswith('_'):
            continue
        n = len(s.index) if hasattr(s, 'index') else len(s)
        print(f"  {name:<30} {n} participants")

    return predictors
