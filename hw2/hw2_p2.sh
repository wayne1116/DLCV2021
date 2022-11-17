# TODO: create shell script for running your GAN model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16lWSpIWXDdNIMoWxSAjk0r6Yz3fj86oV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16lWSpIWXDdNIMoWxSAjk0r6Yz3fj86oV" -O G_2.pth
# Example
python3 p2.py --save_dir $1 
