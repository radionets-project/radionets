import click
from simulations.gaussian_simulations import create_n_bundles


@click.command()
@click.argument('num_bundles', type=int)
@click.argument('bundle_size', type=int)
@click.argument('img_size', type=int)
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-noise', type=bool, required=False)
def main(num_bundles, bundle_size, img_size, out_path, noise=False):
    create_n_bundles(num_bundles, bundle_size, img_size, out_path)


if __name__ == '__main__':
    main()
