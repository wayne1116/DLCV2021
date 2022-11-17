# TODO: create shell script for running your VAE model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EFau0EEfOAaIjWnE_UJ6KhDjr34Tn8H1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EFau0EEfOAaIjWnE_UJ6KhDjr34Tn8H1" -O G_1.pth
# Example
python3 p1.py --save_dir $1 
