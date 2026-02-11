import argparse
import logging

import geopandas as gpd

from cocoa_mapping.finetuning_data.clustering import compute_clusters
from cocoa_mapping.utils.db_utils import does_table_exist, upload_table_to_db, get_full_table
from cocoa_mapping.utils.general_utils import load_env_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_and_upload_clusters(annotations_gdf: gpd.GeoDataFrame,
                                output_table_name: str,
                                output_cluster_table_name: str,
                                buffer_m: float = 320,
                                schema: str = 'cocoa_data',
                                overwrite: bool = False):
    """Compute and upload clusters to the database.

    Args:
        annotations_gdf: The annotations you want to cluster. Should have 'year' column.
        output_table_name: The name of the output table where to save annotations with cluster ids.
            If it already exists, see overwrite parameter.
        output_cluster_table_name: The name of the output table where to save the cluster geometries.
            If it already exists, see overwrite parameter.
        buffer_m: The buffer size in meters. Buffer is required so that we can place the bounding boxes around the annotations.
        schema: The schema of the tables in the database.
        overwrite: Whether to overwrite the existing tables. 
            If False, will raise an error if the tables already exist, except if only annotations table exists and misses cluster ids, which will then be added.
            If True, will overwrite the existing tables. 
    """
    assert 'year' in annotations_gdf.columns, "GeoDataFrame must have a 'year' column"
    if not annotations_gdf.index.is_unique:
        logger.warning("GeoDataFrame has duplicate indices. Resetting the index.")
        annotations_gdf = annotations_gdf.reset_index(drop=True)

    # Check if the output cluster table already exists
    if not overwrite and does_table_exist(output_cluster_table_name, schema):
        raise ValueError(f"Table {output_cluster_table_name} already exists in the database. "
                         "Set overwrite=True to overwrite it.")

     # Compute clusters
    cluster_gdf, annotations_gdf['cluster_id'] = compute_clusters(annotations_gdf, buffer_m=buffer_m)

    # Upload the annotations table to the database
    if overwrite or not does_table_exist(output_table_name, schema):
        upload_table_to_db(annotations_gdf,
                           table_name=output_table_name,
                           schema=schema,
                           if_exists='replace' if overwrite else 'fail')
        logger.info(f"Table {output_table_name} uploaded to the database.")
    else:
        handle_existing_table_no_overwrite(annotations_gdf, output_table_name, schema)

    # Upload the cluster table to the database
    upload_table_to_db(cluster_gdf,
                       table_name=output_cluster_table_name,
                       schema=schema,
                       if_exists='replace' if overwrite else 'fail')
    logger.info(f"Table {output_cluster_table_name} uploaded to the database. We are done here.")


def handle_existing_table_no_overwrite(annotations_gdf: gpd.GeoDataFrame, output_table_name: str, schema: str = 'cocoa_data') -> None:
    """Handle the case where the table already exists in the database and we don't want to overwrite it.

    In this case, we check if they have the same geometries but just cluster ids are missing.
    Yes -> add the cluster ids to the table and upload it to the database.
    No, but geometries and cluster ids are the same -> no action needed.
    No, either geometries or cluster ids are different -> raise an error.

    Args:
        annotations_gdf: The annotations GeoDataFrame.
        output_table_name: The name of the table in the database.
        schema: The schema of the table in the database.
    """
    logger.info(f"Table {output_table_name} already exists in the database. Checking its cluster ids.")
    table_gdf = get_full_table(output_table_name, schema=schema)

    # Check lengths
    if len(table_gdf) != len(annotations_gdf):
        raise ValueError(f"Number of rows in the table {output_table_name} is not the same as the number of rows in the cluster geometry. "
                         "Set overwrite=True to overwrite it.")

    # Check geometry. We do not care about the index, but the order should be the same
    if not table_gdf.geometry.reset_index(drop=True).equals(annotations_gdf.geometry.reset_index(drop=True)):
        raise ValueError(f"Geometry of the table {output_table_name} is not the same as the cluster geometry. "
                         "Set overwrite=True to overwrite it.")

    # At this point, we established that the table has the same geometries as in the cluster_gdf
    # Check if cluster ids are set. If not, amazing, meaning we just need to add them
    if 'cluster_id' not in table_gdf.columns:
        table_gdf['cluster_id'] = annotations_gdf['cluster_id']
        upload_table_to_db(table_gdf,
                           table_name=output_table_name,
                           schema=schema,
                           if_exists='replace')
        logger.info(f"Table {output_table_name} uploaded to the database with cluster ids.")
        return

    # Check if the cluster ids are the same
    if not (table_gdf['cluster_id'] == annotations_gdf['cluster_id']).all():
        raise ValueError(f"Cluster ids are not the same in the table {output_table_name}. "
                         "Set overwrite=True to overwrite it.")

    logger.info(f"Table {output_table_name} already exists in the database and has the same geometries and cluster ids as the annotations. No action needed.")
    return


if __name__ == "__main__":
    # Load the annotations
    parser = argparse.ArgumentParser(description=(
        "Compute spatial clusters from annotated geometries and upload both the clustered "
        "annotations and cluster geometries to a PostGIS database. The script accepts either "
        "a local GeoJSON file or an existing database table as input, computes clusters based "
        "on a specified buffer distance, and writes two output tables:\n"
        "1) the annotations table with a new 'cluster_id' column, and\n"
        "2) a separate cluster geometry table.\n\n"
        "If existing tables already exist in the target schema, you can control overwrite "
        "behavior with the --overwrite flag.")
    )

    # Required arguments
    parser.add_argument('-ag', '--annotations-geojson', type=str, default=None,
                        help="The path to the annotations GeoJSON file.")
    parser.add_argument('-at', '--annotations-table', type=str, default=None,
                        help="If annotations-geojson not provided, the name of the table in the database.")
    parser.add_argument('-ot', '--output-table-name', type=str, required=True,
                        help="The name of the output table in the database with cluster ids.")
    parser.add_argument('-oc', '--output-cluster-table-name', type=str, required=True,
                        help="The name of the output cluster table in the database.")

    # Optional arguments
    parser.add_argument('-b', '--buffer-m', type=float, default=320,
                        help="The buffer in meters to use for clustering.")
    parser.add_argument('-s', '--schema', type=str, default='cocoa_data',
                        help="The schema of the table in the database.")
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help="If False, will avoid overwriting the existing tables "
                        "Note: If true and output table exists but misses cluster ids, they will be added.")
    args = parser.parse_args()

    load_env_file()

    # Load the annotations
    if args.annotations_geojson is not None:
        annotations_gdf = gpd.read_file(args.annotations_geojson)
    elif args.annotations_table is not None:
        annotations_gdf = get_full_table(args.annotations_table, schema=args.schema)
    else:
        raise ValueError("Either annotations-geojson or annotations-table must be provided.")

    compute_and_upload_clusters(annotations_gdf,
                                output_table_name=args.output_table_name,
                                output_cluster_table_name=args.output_cluster_table_name,
                                buffer_m=args.buffer_m,
                                schema=args.schema,
                                overwrite=args.overwrite)
