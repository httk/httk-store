"""DuckDB and Arrow must compose without changing optional pandas discovery."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("pandas_available", [False, True])
def test_duckdb_and_arrow_preserve_pandas_discovery(pandas_available):
    """Scalar binding and Arrow arrays work with absent or installed pandas."""
    pytest.importorskip("duckdb_engine")
    pytest.importorskip("pyarrow")
    if pandas_available:
        pytest.importorskip("pandas")
    code = r"""
import importlib.util
import sys

if sys.argv[1] == "False":
    original_finders = list(sys.meta_path)
    class WithoutPandas:
        def find_distributions(self, context):
            for finder in original_finders:
                if hasattr(finder, "find_distributions"):
                    yield from finder.find_distributions(context)

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "pandas" or fullname.startswith("pandas."):
                return None
            for finder in original_finders:
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    return spec
            return None
    sys.meta_path = [WithoutPandas()]
    assert importlib.util.find_spec("pandas") is None
else:
    import pandas
    original_pandas = pandas

import sqlalchemy
from httk.store.backend.sql import Backend

backend = Backend.duckdb()
try:
    with backend.engine.begin() as connection:
        connection.execute(sqlalchemy.text("CREATE TABLE probe (a INTEGER)"))
        connection.execute(sqlalchemy.text("INSERT INTO probe VALUES (:a)"), [{"a": i} for i in range(3)])
        assert connection.execute(sqlalchemy.text("SELECT SUM(a) FROM probe")).scalar_one() == 3
        assert connection.execute(sqlalchemy.text(
            "SELECT httk_fraction_scaled_equal('1/3', '2', '2/3', '1')"
        )).scalar_one() is True
    import pyarrow as pa
    assert pa.array([1, 2]).to_pylist() == [1, 2]
    if sys.argv[1] == "False":
        assert "pandas" not in sys.modules
        assert importlib.util.find_spec("pandas") is None
    else:
        assert sys.modules["pandas"] is original_pandas
finally:
    backend.dispose()
"""
    result = subprocess.run([sys.executable, "-c", code, str(pandas_available)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
