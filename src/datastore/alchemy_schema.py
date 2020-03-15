import pymysql
from sqlalchemy.dialects.mysql import LONGBLOB

pymysql.install_as_MySQLdb()
from sqlalchemy import Column, String, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datastore.sql_path import sydney_mysql_path
from datastore.table_names import *

Base = declarative_base()


class KeyValueBase(object):
    key = Column('key', String(256), primary_key=True, index=True)
    value = Column('value', LONGBLOB)


class TestTable(Base, KeyValueBase):
    __tablename__ = "test_table"


class RawCluewebDocTable(Base, KeyValueBase):
    __tablename__ = RawCluewebDoc


class CleanedCluewebDocTable(Base, KeyValueBase):
    __tablename__ = CleanedCluewebDoc


class BertTokenizedCluewebDocTable(Base, KeyValueBase):
    __tablename__ = BertTokenizedCluewebDoc


class TokenizedCluewebDocTable(Base, KeyValueBase):
    __tablename__ = TokenizedCluewebDoc


class CluewebDocTFTable(Base, KeyValueBase):
    __tablename__ = CluewebDocTF


from sqlalchemy import create_engine

engine = create_engine(sydney_mysql_path)


def create_table():
    Base.metadata.create_all(engine)


all_tables = [TestTable,
              RawCluewebDocTable,
              CleanedCluewebDocTable,
              BertTokenizedCluewebDocTable,
              TokenizedCluewebDocTable,
              CluewebDocTFTable,
              ]


table_class_by_name = {table.__tablename__: table for table in all_tables}


def index_all():
    for table in all_tables:
        index_table(table)


def index_table(table):
    index_ = Index(table.__tablename__ + '__index', table.key)
    index_.create(bind=engine)


Session = sessionmaker(bind=engine)
