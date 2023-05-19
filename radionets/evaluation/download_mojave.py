import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import click
import pandas as pd
import requests
from astropy.io import fits
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_mojave(save_path: str, download: bool = True, cleanup: bool = True):
    """Download all data from the MOJAVE dataset. Sources with no .icn.fits.gz file or
    no redshift can be removed. The redshift and velocity (speed) will be written into
    the header of the fits file for faster access, if cleanup is True.
    The download takes several hours (~10h) as well as the cleanup (~2h).

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
                "-r",
                "-l",
                "inf",
                "-nc",
                # "-m",  # --mirror equivalent to "-r -N -l inf --no-remove-listing"
                "-q",  # --quiet no output
                "-np",  # --no-parent don't download from parent directories
                "-e",  # --execute (the following) command
                "robots=off",  # robots.txt don't allow download (apparently)
                "-A",  # --accept only accept the following pattern
                ".icn.fits.gz",  # ending of the desired files
                "-P",  # --directory-prefix directory to save data
                save_path,
                url,
            ]
        )
    click.echo(f"{current_time()} - Download finished \n")

    if cleanup:
        click.echo(f"{current_time()} - Starting clean up \n")
        if os.path.isdir(save_path + "www.cv.nrao.edu"):
            click.echo(f"{current_time()} - Remove unnecessary parent folders \n")
            shutil.copytree(
                save_path + "www.cv.nrao.edu/2cmVLBA/data/",
                save_path,
                dirs_exist_ok=True,
            )
            subprocess.run(["rm", "-r", save_path + "www.cv.nrao.edu"])

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
            redshift, vs, v_uncs = get_properties(source)
            if redshift and vs and v_uncs:
                for file in data_files:
                    with fits.open(file, mode="update") as hdul:
                        hdr = hdul[0].header
                        hdr.set(
                            "REDSHIFT", redshift, "Optical redshift", after="OBJECT"
                        )
                        for i, (v, v_unc) in enumerate(zip(vs, v_uncs)):
                            hdr.set(
                                f"V{i+1}",
                                float(v),
                                "Jet velocity (speed)",
                                after="REDSHIFT",
                            )
                            hdr.set(
                                f"V_UNC{i+1}",
                                float(v_unc),
                                "Jet velocity (speed) uncertainty",
                                after=f"V{i+1}",
                            )
                        hdul.close(output_verify="silentfix+ignore")
            else:
                subprocess.run(["rm", "-r", source_path])
                source_list.remove(source)
                click.echo(
                    f"{current_time()} - Scoure {source}: z = {redshift}, v = {vs}, v_unc = {v_uncs} -> Removed"
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
    v: list
        list of velocity (speed) from all components
    v_unc: list
        list of velocity (speed) uncertainty from all components
    """
    url_redshift = "https://www.cv.nrao.edu/MOJAVE/sourcepages/" + source + ".shtml"

    try:
        page = urlopen(url_redshift)
    except HTTPError:
        return False, False, False

    html_bytes = page.read()
    soup = BeautifulSoup(html_bytes, features="html.parser")

    redshift = soup.find("td", string="Redshift:").find_next_sibling("td").text
    try:
        redshift = float(redshift.split()[0])
    except ValueError:
        redshift = False

    url_table = "https://www.cv.nrao.edu/MOJAVE/velocitytableXVIII.html"

    html = requests.get(url_table).content
    df_list = pd.read_html(html)

    df = df_list[-1]
    velocities = df[df["B1950 Name"] == source]["Speed (c)"].values
    v = [v.split("&")[0] for v in velocities if type(v) == str]
    v_unc = [v_unc.split("n")[1] for v_unc in velocities if type(v_unc) == str]
    return redshift, v, v_unc


def current_time():
    return time.strftime("%H:%M:%S", time.localtime())
