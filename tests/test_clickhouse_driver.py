"""Binary-query compatibility with the installed ClickHouse Connect DBAPI."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import sqlalchemy

pytest.importorskip("clickhouse_connect")

from clickhouse_connect.dbapi.cursor import Cursor
from clickhouse_connect.driver.query import QueryContext

from httk.store.backend.clickhouse import support


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("parameters", [None, {"value": 1}])
@pytest.mark.parametrize("pyformat_encoded", [False, True])
def test_driver_query_formats_preserve_options_and_percent_escaping(monkeypatch, binary, parameters, pyformat_encoded):
    original_execute = inspect.unwrap(Cursor.execute)
    monkeypatch.setattr(Cursor, "execute", original_execute)
    monkeypatch.setattr(Cursor, "_httk_binary_query_formats", False, raising=False)
    native = "query_formats" in inspect.signature(original_execute).parameters
    assert support._install_binary_query_format_hook() is native
    assert (Cursor.execute is original_execute) is native
    installed_execute = Cursor.execute
    assert support._install_binary_query_format_hook() is native
    assert Cursor.execute is installed_execute

    calls = []

    class Client:
        def query(self, query, parameters=None, **kwargs):
            calls.append((query, parameters, kwargs))
            return SimpleNamespace(
                result_set=[(b"\x00\xff",)],
                summary={},
                column_names=["payload"],
                column_types=[SimpleNamespace(name="String")],
            )

    cursor = Cursor(Client())
    engine = sqlalchemy.create_engine("clickhousedb://default@localhost/default")
    try:
        support._install_binary_query_event(engine, native_query_formats=native)
        connection_formats = {"*": "str", "UUID": "string"}
        statement_formats = {"Date": "int", "String*": "str", "String": "str"}
        connection_settings = {"max_threads": 1}
        statement_settings = {"max_block_size": 2}
        originals = copy.deepcopy((connection_formats, statement_formats, connection_settings, statement_settings))
        statement = sqlalchemy.select(
            sqlalchemy.literal_column("payload", type_=sqlalchemy.LargeBinary() if binary else sqlalchemy.String())
        ).execution_options(query_formats=statement_formats, settings=statement_settings)
        options = sqlalchemy.util.immutabledict(query_formats=connection_formats, settings=connection_settings)
        context = SimpleNamespace(
            invoked_statement=statement,
            execution_options=options,
            compiled=SimpleNamespace(preparer=SimpleNamespace(_double_percents=pyformat_encoded)),
        )
        operation = "SELECT '50%%', payload"
        engine.dispatch.before_cursor_execute(None, cursor, operation, parameters, context, False)
        engine.dialect.do_execute(cursor, operation, parameters, context)
        query, received_parameters, kwargs = calls.pop()
        assert not calls
        expected_query = operation if parameters or native and not pyformat_encoded else "SELECT '50%', payload"
        assert query == expected_query
        assert received_parameters == parameters
        assert kwargs["settings"] == {"max_threads": 1, "max_block_size": 2}
        formats = kwargs.get("query_formats")
        if binary:
            assert QueryContext(query_formats=formats).active_fmt("String") == "bytes"
            if native:
                assert formats == {"String": "bytes", "Date": "int", "String*": "str", "*": "str", "UUID": "string"}
                assert list(formats)[0] == "String"
        elif native:
            assert formats == {**statement_formats, **connection_formats}
            assert context.invoked_statement is statement
            assert context.execution_options is options
        else:
            assert formats is None
        assert tuple(cursor.fetchone()) == (b"\x00\xff",)
        assert (connection_formats, statement_formats, connection_settings, statement_settings) == originals
        assert statement.get_execution_options()["query_formats"] is statement_formats
    finally:
        engine.dispose()
