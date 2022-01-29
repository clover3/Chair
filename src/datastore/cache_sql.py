import os.path

import pymysql

pymysql.install_as_MySQLdb()
from sqlalchemy import Column, String, Index, REAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()


class KeyValueBase(object):
    key = Column('key', String(), primary_key=True, index=True, sqlite_on_conflict_not_null='IGNORE')
    value = Column('value', REAL)


class CacheTable(Base, KeyValueBase):
    __tablename__ = "cache"


def get_engine_from_sqlite_path(sqlite_path):
    sqlite_path = 'sqlite:///' + sqlite_path
    engine = create_engine(sqlite_path)
    return engine


def index_table(table, engine):
    index_ = Index(table.__tablename__ + '__index', table.key)
    index_.create(bind=engine)


def run_index_table(sqlite_path):
    sqlite_path = os.path.abspath(sqlite_path)
    index_table(CacheTable, get_engine_from_sqlite_path(sqlite_path))


def main():
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    main()