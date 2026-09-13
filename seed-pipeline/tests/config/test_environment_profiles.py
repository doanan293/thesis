from seed_pipeline.config.environment import parse_env_file
from seed_pipeline.config.paths import PROJECT_ROOT


def test_env_example_documents_numbered_kaggle_profiles():
    values = parse_env_file(PROJECT_ROOT / ".env.example")
    assert values["KAGGLE_ACCOUNT_DEFAULT"] == "acc1"
    assert values["KAGGLE_SHARED_OWNER"] == "your_primary_kaggle_username_here"
    assert values["KAGGLE_ACC1_USERNAME"] == "your_primary_kaggle_username_here"
    assert values["KAGGLE_ACC1_API_TOKEN"] == "your_acc1_api_token_here"
    assert values["KAGGLE_ACC2_USERNAME"] == "your_secondary_kaggle_username_here"
    assert values["KAGGLE_ACC2_API_TOKEN"] == "your_acc2_api_token_here"
