import pytest


@pytest.fixture(scope="session")
def h():
    from consulted_bridges.load import load

    return load("tiny")


@pytest.fixture(scope="session")
def pool():
    from consulted_bridges.silent_dissent import facts as F

    return F.load()


@pytest.fixture(scope="session")
def mozart(pool):
    from consulted_bridges.silent_dissent import facts as F

    return F.find(pool, "Wolfgang Amadeus Mozart")
