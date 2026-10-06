from neo4j import Driver, GraphDatabase

from graphrag.config import Settings


class Neo4jStore:
    def __init__(self, settings: Settings) -> None:
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )

    def ping(self) -> None:
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()
