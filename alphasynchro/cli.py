#!python
'''Module to control alphasynchro from command-line interface.'''


# external
import click

# local
import alphasynchro
import alphasynchro.io.logging


@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
)
@click.pass_context
@click.version_option(alphasynchro.__version__, "-v", "--version")
def run(ctx, **kwargs) -> None:
    name = f"alphasynchro { alphasynchro.__version__}"
    click.echo("*" * (len(name) + 4))
    click.echo(f"* {name} *")
    click.echo("*" * (len(name) + 4))
    alphasynchro.io.logging.set_logger()
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("gui", help="Start graphical user interface.")
def gui() -> None:
    import alphasynchro.gui
    alphasynchro.gui.run()


@run.command(
    "create_spectra",
    help='Creates ms2 spectra from a cluster hdf file.',
    no_args_is_help=True,
)
@click.option(
    "--analysis_file_name",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
    help="A file where to store the results of this analysis.",
)
@click.option(
    "--cluster_file_name",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="A peakpicker preprocessed .hdf file.",
)
@click.option(
    "--threads",
    type=int,
    default=31,
    help="Number of threads (negative is how many to leave available, 0 means all)",
    show_default=True,
)
@click.option(
    "--max_rt_weight",
    type=float,
    default=0.5,
    help="The maximum accepted ks-distance for rt.",
    show_default=True,
)
@click.option(
    "--max_im_weight",
    type=float,
    default=0.5,
    help="The maximum accepted ks-distance for im.",
    show_default=True,
)
@click.option(
    "--max_frame_weight",
    type=float,
    default=0.5,
    help="The maximum accepted ks-distance for frames.",
    show_default=True,
)
@click.option(
    "--unique_transitions_only",
    is_flag=True,
    default=False,
    help="Use only the best unique transitions.",
    show_default=True,
)
@click.option(
    "--min_fragment_size",
    type=int,
    default=1,
    help="The minimum fragment size.",
    show_default=True,
)
@click.option(
    "--diapasef",
    is_flag=True,
    default=False,
    help="Use regular diapasef rather than synchropasef.",
    show_default=True,
)
def create_spectra(
    analysis_file_name: str,
    cluster_file_name: str,
    threads: int,
    max_rt_weight: float,
    max_im_weight: float,
    max_frame_weight: float,
    unique_transitions_only: bool,
    min_fragment_size: int,
    diapasef: bool,
) -> None:
    import alphasynchro.algorithms.pipeline
    import alphasynchro.performance.multithreading
    alphasynchro.io.logging.show_platform_info()
    alphasynchro.io.logging.show_python_info()
    alphasynchro.performance.multithreading.set_threads(threads)
    pipeline = alphasynchro.algorithms.pipeline.Pipeline(
        analysis_file_name,
        overwrite=True
    )
    pipeline.run(
        cluster_file_name=cluster_file_name,
        max_rt_weight=max_rt_weight,
        max_im_weight=max_im_weight,
        max_frame_weight=max_frame_weight,
        unique_transitions_only=unique_transitions_only,
        min_fragment_size=min_fragment_size,
        diapasef=diapasef,
    )

@run.command(
    "write_mgf",
    help='Creates ms2 spectra from a cluster hdf file.',
    no_args_is_help=True,
)
@click.option(
    "--analysis_file_name",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
    help="A file where to store the results of this analysis.",
)
@click.option(
    "--spectra_file_name",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
def write_mgf(
    analysis_file_name: str,
    spectra_file_name: str,
) -> None:
    import alphasynchro.algorithms.pipeline
    alphasynchro.io.logging.show_platform_info()
    alphasynchro.io.logging.show_python_info()
    pipeline = alphasynchro.algorithms.pipeline.Pipeline(
        analysis_file_name
    )
    pipeline.write_ms2_spectra(spectra_file_name)


if __name__ == "__main__":
    run()
