import os
from geopandas import gpd
import psycopg2
from sqlalchemy import create_engine
import sqlalchemy
from shapely import wkb


def does_table_exist(table_name: str, schema: str = 'cocoa_data') -> bool:
    """Check if a table exists in the database."""
    conn = _get_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT to_regclass('{schema}.{table_name}')")
        return cur.fetchone()[0] is not None


def upload_table_to_db(gdf: gpd.GeoDataFrame,
                       table_name: str,
                       schema: str = 'cocoa_data',
                       if_exists: str = 'fail') -> None:
    """Upload a GeoDataFrame to the database."""
    engine = _get_engine()
    gdf.to_postgis(table_name, engine, schema=schema, if_exists=if_exists)


def _get_connection():
    """Get a connection to the database."""
    database = os.environ["GEO_DB_NAME"]
    user = os.environ["GEO_DB_USER"]
    password = os.environ["GEO_DB_PASSWORD"]
    host = os.environ["GEO_DB_HOST"]
    port = os.environ["GEO_DB_PORT"]
    return psycopg2.connect(
        dbname=database,
        user=user,
        password=password,
        host=host,
        port=port
    )


def _get_engine() -> sqlalchemy.engine.Engine:
    """Get the engine for the database."""
    # Set default values from environment variables if not provided
    database = os.environ["GEO_DB_NAME"]
    user = os.environ["GEO_DB_USER"]
    password = os.environ["GEO_DB_PASSWORD"]
    host = os.environ["GEO_DB_HOST"]
    port = os.environ["GEO_DB_PORT"]
    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    )
    return engine


def get_cursor():
    """Get a cursor for the database."""
    engine = _get_engine()
    return engine.connect()


def get_full_table(
        table_name: str,
        sql_filter: str = '1=1',
        schema: str = "cocoa_data",
) -> gpd.GeoDataFrame:
    """
    Get full table from the database.

    Args:
        table_name (str): The name of the table to query.
        sql_filter (str): The filter to apply to the database table. Default is '1=1' which is no filter.
        schema (str): The name of the schema. Default is "cocoa_data".
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the full table
    """
    conn = _get_connection()
    with conn.cursor() as cur:
        if schema is not None:
            query = f"""
            SELECT *
            FROM \"{schema}\".\"{table_name}\"
            WHERE {sql_filter};
            """
        else:
            query = f"""
            SELECT *
            FROM \"{table_name}\"
            WHERE {sql_filter};
        """

        # Execute the query but put as many %s as the number of columns
        cur.execute(query)
        rows = cur.fetchall()

        # Get column names
        col_names = [desc[0] for desc in cur.description]
        gdf = gpd.GeoDataFrame(rows, columns=col_names)

    # If the query returns no results, raise an error
    if gdf.empty:
        raise RuntimeError(
            f"No data found in table {table_name}"
        )

    # If the gdf has a geometry column, convert WKB to geometry
    if 'geometry' in gdf.columns:
        gdf['geometry'] = gdf['geometry'].apply(wkb.loads)
        gdf.set_geometry('geometry', inplace=True)
    elif 'geom' in gdf.columns:
        gdf['geometry'] = gdf['geom'].apply(wkb.loads)
        gdf.set_geometry('geometry', inplace=True)
        gdf.drop(columns=['geom'], inplace=True)

    # If gdf crs is not set, set it to epsg 4326
    if gdf.crs is None:
        gdf.crs = "EPSG:4326"

    return gdf
