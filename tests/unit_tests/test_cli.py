# external
import click.testing

# local
import alphasynchro.cli


def test_runner():
    runner = click.testing.CliRunner()
    result = runner.invoke(alphasynchro.cli.run, [])
    assert result.exit_code == 0


def test_create_spectra():
    runner = click.testing.CliRunner()
    result = runner.invoke(alphasynchro.cli.run, ["create_spectra"])
    assert result.exit_code == 0


def test_write_mgf():
    runner = click.testing.CliRunner()
    result = runner.invoke(alphasynchro.cli.run, ["write_mgf"])
    assert result.exit_code == 0
