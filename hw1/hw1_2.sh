# Download trained model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PCC267N29qiYSE-Gue2Kl_rTA-x0pKVi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PCC267N29qiYSE-Gue2Kl_rTA-x0pKVi" -O best_seg.pth

# execuate the program
python3 hw1_2.py --img_dir $1 --save_dir $2
