from pydantic import BaseModel


class DatasetSetup(BaseModel):
    """This class defines name and version of a dataset in the database."""
    table_name: str
    """Name of the table in the database."""
    version: str
    """Version of the table in the database."""

    def __repr__(self) -> str:
        return f"DatasetSetup(table_name='{self.table_name}', version='{self.version}')"

    def pretty_name(self) -> str:
        """Name in the format 'table_name version'."""
        return f"{self.table_name} {self.version}"


class EvaluationSetup(BaseModel):
    """This class defines table names and versions for a specific evaluation setup."""
    name: str
    """Name of the evaluation setup."""
    test_tiles: DatasetSetup
    """Name and version of the test tiles table."""
    countries: dict[str, DatasetSetup]
    """Name and version of tables to use for each country."""

    def support_country(self, country: str) -> bool:
        """Check if the evaluation setup supports a given country."""
        return country in self.countries


THESIS_EVALUATION_SETUP = EvaluationSetup(
    name='thesis',
    test_tiles=DatasetSetup(table_name='sentinel_2_test_tiles', version='v1'),
    countries={
        'Ivory Coast': DatasetSetup(table_name='cote_divoire_unified_dataset_v1', version='v1'),
        'Ghana': DatasetSetup(table_name='ghana_unified_dataset_v1', version='v1'),
        'Nigeria': DatasetSetup(table_name='nigeria_unified_dataset_v2', version='v2'),
        'Cameroon': DatasetSetup(table_name='cameroon_unified_dataset_v1', version='v1'),
    }
)


ALL_EVALUATION_SETUPS: list[EvaluationSetup] = [THESIS_EVALUATION_SETUP]
"""All evaluation setups. When defining a new evaluation setup, add it to this list."""


ALL_SUPPORTED_COUNTRIES: set[str] = {country for setup in ALL_EVALUATION_SETUPS for country in setup.countries.keys()}
"""All supported countries."""


EVALUATION_SETUPS_REGISTRY: dict[str, EvaluationSetup] = {setup.name: setup for setup in ALL_EVALUATION_SETUPS}
"""Map from evaluation setup name to evaluation setup."""


def get_evaluation_setup(evaluation_setup_name: str) -> EvaluationSetup:
    """Get an evaluation setup by name.

    Args:
        evaluation_setup_name: The name of the evaluation setup, e.g. 'thesis'. It defines which tables to use.

    Returns:
        The evaluation setup defining name and version of the tables to use for the evaluation.
    """
    if evaluation_setup_name not in EVALUATION_SETUPS_REGISTRY:
        available_setups = list(EVALUATION_SETUPS_REGISTRY.keys())
        raise ValueError(f"Evaluation setup {evaluation_setup_name} not found. Please choose from: {available_setups}")
    return EVALUATION_SETUPS_REGISTRY[evaluation_setup_name]


def get_test_tiles_setup(evaluation_setup_name: str) -> DatasetSetup:
    """Get the name and the version of the test tiles table for an evaluation setup.

    Args:
        evaluation_setup_name: The name of the evaluation setup, e.g. 'thesis'. It defines which tables to use.

    Returns:
        The test tiles setup (table name and version).
    """
    return get_evaluation_setup(evaluation_setup_name).test_tiles


def get_country_dataset_setup(evaluation_setup_name: str, country: str) -> DatasetSetup:
    """Get the country dataset setup for an evaluation setup.

    Args:
        evaluation_setup_name: The name of the evaluation setup.
        country: The country to get the dataset for.

    Returns:
        The setup (table name and version) for the country.
    """
    evaluation_setup = get_evaluation_setup(evaluation_setup_name)
    if country not in evaluation_setup.countries:
        raise ValueError(f"Country {country} not found in evaluation setup {evaluation_setup_name}. Please choose from: {[country for country in evaluation_setup.countries.keys()]}")
    return evaluation_setup.countries[country]
