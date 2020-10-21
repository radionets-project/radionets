#!/bin/bash

# setup script to install texlive and add to path
texlive_year="2020"

sudo apt-get -qq update
export PATH=/tmp/texlive/bin/x86_64-linux:$PATH

if ! command -v lualatex > /dev/null; then
    echo "Texlive not installed"
    echo "Downloading texlive and installing"
    # wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
    curl -L http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz | tar xz
    TEXLIVE_INSTALL_PREFIX=~/tmp/texlive ./install-tl-*/install-tl
    echo | <I>
    # tar -xzf install-tl-unx.tar.gz
    # ./install-tl-*/install-tl --profile=./utilities/texlive.profile
    echo "Finished install TexLive"
fi

echo "Now updating TexLive"
# update texlive
luatex
tlmgr option -- autobackup 0
tlmgr update --self --all --no-auto-install

echo "Finished updating TexLive"
