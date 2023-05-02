import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import click
from astropy.io import fits
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_mojave(save_path: str, download: bool = True, cleanup: bool = True):
    """Download all data from the MOJAVE dataset. Sources with no .icn.fits.gz file or
    no redshift can be removed. The redshift and speed will be written into the header
    of the fits file for faster access, if cleanup is True.
    The download takes several hours (~8h) as well as the cleanup (~2h).

    Parameters
    ----------
    save_path: str
        path to directory where to save the files
    download: bool
        possibility to skip download
    cleanup: bool
        remove sources, which cannot be analysed
    """
    click.echo(
        f"{current_time()} - Starting script to download and clean MOJAVE files\n"
    )
    url = "https://www.cv.nrao.edu/2cmVLBA/data/"

    if save_path[-1:] != "/":
        save_path = save_path + "/"

    if download:
        click.echo(f"{current_time()} - Downloading folder structure and files")
        subprocess.run(
            [
                "wget",
                "-m",
                "-q",
                "-np",
                "-e",
                "robots=off",
                "-A",
                ".icn.fits.gz",
                "-P",
                save_path,
                url,
            ]
        )
    click.echo(f"{current_time()} - Download finished \n")

    if os.path.isdir(save_path + "www.cv.nrao.edu"):
        click.echo(f"{current_time()} - Remove unnecessary parent folders \n")
        shutil.copytree(
            save_path + "www.cv.nrao.edu/2cmVLBA/data/", save_path, dirs_exist_ok=True
        )
        subprocess.run(["rm", "-r", save_path + "www.cv.nrao.edu"])

    if cleanup:
        click.echo(f"{current_time()} - Starting clean up \n")
        source_list = sorted(next(os.walk(save_path))[1])

        click.echo(f"{current_time()} - Removing folders which are not a source \n")
        for item in tqdm(source_list):
            if not item[:4].isnumeric():
                subprocess.run(["rm", "-r", save_path + item])
                click.echo(f"Removed folder: {item}")
        source_list = [item for item in source_list if item[:4].isnumeric()]

        click.echo(
            f"\n{current_time()} - Remove sources with one or less images, "
            + "invalid redshift or invalid speed \n"
        )
        for source in tqdm(reversed(source_list), total=len(source_list)):
            source_path = save_path + source
            file_paths = [x for x in Path(source_path).glob("*/*")]
            data_files = [
                path for path in file_paths if re.findall(".icn.fits.gz", path.name)
            ]
            if len(data_files) <= 1:
                subprocess.run(["rm", "-r", source_path])
                source_list.remove(source)
                click.echo(
                    f"{current_time()} - {len(data_files)} images for source {source}: Removed"
                )
                continue

            redshift, speed, speed_unc, n_features = get_properties(source)
            if redshift and speed and speed_unc:
                for file in data_files:
                    with fits.open(file, mode="update") as hdul:
                        hdr = hdul[0].header
                        hdr.set(
                            "REDSHIFT", redshift, "Optical redshift", after="OBJECT"
                        )
                        hdr.set("SPEED", speed, "Jet speed", after="REDSHIFT")
                        hdr.set(
                            "SPEEDUNC",
                            speed_unc,
                            "Jet speed uncertainty",
                            after="SPEED",
                        )
                        hdr.set(
                            "N_FEAT",
                            n_features,
                            "Velocity based on n moving features",
                            after="SPEEDUNC",
                        )
                        hdul.close(output_verify="silentfix+ignore")
            else:
                subprocess.run(["rm", "-r", source_path])
                source_list.remove(source)
                click.echo(
                    f"{current_time()} - Scoure {source}: z = {redshift}, v = {speed} -> Removed"
                )
                continue

        click.echo(f"\n{current_time()} - Clean up finished")


def get_properties(source: str):
    """Get properties of source from MOJAVE webpage.
    If webpage does not exist or property is unknown, return False, because
    fits-files do not support NaN.

    Parameters
    ----------
    source: str
        B1950 name of source

    Returns
    -------
    redshift: float
        Redshift of the object
    speed: float
        speed of the object
    speed_unc: float
        uncertainty of the speed of the object
    n_features: int
        velocity based on n moving features
    """
    url = "https://www.cv.nrao.edu/MOJAVE/sourcepages/" + source + ".shtml"

    try:
        page = urlopen(url)
    except HTTPError:
        return False, False, False, False

    html_bytes = page.read()
    soup = BeautifulSoup(html_bytes, features="html.parser")

    redshift = soup.find("td", string="Redshift:").find_next_sibling("td").text
    try:
        redshift = float(redshift.split()[0])
    except ValueError:
        redshift = False

    speed_str = soup.find("td", string="Jet Speed:").find_next_sibling("td").text
    speed_list = speed_str.split()

    try:
        c_exists = speed_list[9] == "c"
    except IndexError:
        c_exists = False
    if c_exists:
        speed = float(speed_list[6])
        speed_unc = float(speed_list[8])
    else:
        speed, speed_unc = False, False

    try:
        idx = speed_list.index("on") + 1
        n_features = int(speed_list[idx])
    except ValueError:
        n_features = False

    return redshift, speed, speed_unc, n_features


def current_time():
    return time.strftime("%H:%M:%S", time.localtime())
