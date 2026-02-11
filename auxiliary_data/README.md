## World Administrative Boundaries (Countries)
- Source file: `auxiliry_data/world_administrative_boundaries_countries.geojson`
- Downloaded from: https://public.opendatasoft.com/explore/assets/world-administrative-boundaries-countries/export/
- License: Open Government Licence v3.0 — http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
- Adjustments: kept only `english_short` and `geometry`, renamed `english_short` to `country`, and renamed `Cote d'Ivoire` to `Ivory Coast` for backward compatibility with experiments.

## Sentinel-2 Grid
- Source file: `auxiliry_data/sentinel_2_grid.geojson`
- Downloaded from: https://zenodo.org/records/10998972. All MultiPolygon were then converted into Polygon by taking the first (and only) Polygon from the MultiPolygon.
- License: Creative Commons Attribution 4.0 International
- Citation: Tsironis, V. (2024). Sentinel-2 Tiling Grid (WGS84) in geojson format. Zenodo. https://doi.org/10.5281/zenodo.10998972

## Sentinel-2 test tiles
- Source file: `auxiliry_data/sentinel_2_grid.geojson`. Test tiles for Ivory Coast, Ghana, Nigeria, and Cameroon. 
- Derived from the `auxiliry_data/sentinel_2_grid.geojson` mentioned above.


