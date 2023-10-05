#external
import pytest
import click.testing

# local
import alphasynchro.cli


def test_passes():
    with pytest.raises(NotImplementedError):
        runner = click.testing.CliRunner()
        result = runner.invoke(alphasynchro.cli.run, ["gui"])
        raise result.exception
