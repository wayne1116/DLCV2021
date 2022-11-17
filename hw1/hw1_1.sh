# Download trained model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nBzr5fwE0l9yAA-r8mWDuNhQOIWLg19h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nBzr5fwE0l9yAA-r8mWDuNhQOIWLg19h" -O best_acc.pth

# execuate the program
python3 hw1_1.py --img_dir $1 --save_csv $2
