from alembic.config import Config

SCRIPT_LOCATION = "pharma_agent.infrastructure.persistence.postgres:migrations"


def alembic_config(dsn: str | None = None) -> Config:
    """Alembic config that works from an installed package; `dsn` overrides settings."""
    config = Config()
    config.set_main_option("script_location", SCRIPT_LOCATION)
    config.attributes["dsn"] = dsn
    return config
